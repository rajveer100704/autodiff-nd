use crate::compiler::NodeId;
use ndarray::ArrayD;
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, RwLock,
    },
    cell::Cell,
};

thread_local! {
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|flag| flag.get())
}

pub fn no_grad<F: FnOnce() -> R, R>(f: F) -> R {
    GRAD_ENABLED.with(|flag| {
        let prev = flag.get();
        flag.set(false);
        let result = f();
        flag.set(prev);
        result
    })
}


// ---------------------------------------------------------------------------
// TensorInner: the shared node in the computation graph.
// ---------------------------------------------------------------------------
pub struct TensorInner {
    pub data: RwLock<ArrayD<f64>>,
    pub grad: RwLock<ArrayD<f64>>,
    pub requires_grad: AtomicBool,
    pub grad_fn: Mutex<Option<Arc<dyn BackwardFn + Send + Sync>>>,
}

#[derive(Clone)]
pub enum TensorMode {
    Eager(Arc<TensorInner>),
    Lazy(NodeId),
}

/// The public-facing tensor handle. Cheaply cloneable.
/// Now supports both Eager execution and Lazy graph tracing.
#[derive(Clone)]
pub struct Tensor {
    pub mode: TensorMode,
    pub shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// BackwardFn trait — Send + Sync required for Rayon / Arc storage.
// ---------------------------------------------------------------------------
pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>>;
    fn parents(&self) -> Vec<Tensor>;
    fn op_name(&self) -> &'static str {
        "unknown"
    }
}

// ---------------------------------------------------------------------------
// Custom Autograd: Context and CustomFunction
// ---------------------------------------------------------------------------
pub struct Context {
    pub saved_tensors: Vec<ArrayD<f64>>,
}

impl Context {
    pub fn new() -> Self {
        Self { saved_tensors: Vec::new() }
    }
    pub fn save_for_backward(&mut self, tensor: ArrayD<f64>) {
        self.saved_tensors.push(tensor);
    }
}

pub trait CustomFunction: Send + Sync {
    fn forward(&self, inputs: &[ArrayD<f64>], ctx: &mut Context) -> Vec<ArrayD<f64>>;
    fn backward(&self, ctx: &Context, grad_outputs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
}

pub struct CustomBackward {
    pub func: Arc<dyn CustomFunction>,
    pub ctx: Context,
    pub parents: Vec<Tensor>,
}

impl BackwardFn for CustomBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        self.func.backward(&self.ctx, &[grad_output])
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "CustomFunction"
    }
}


// ---------------------------------------------------------------------------
// reduce_grad: broadcast gradient reduction back to target shape.
// ---------------------------------------------------------------------------
pub fn reduce_grad(grad: ArrayD<f64>, target_shape: &[usize]) -> ArrayD<f64> {
    let mut res = grad;
    let mut target = target_shape.to_vec();
    while target.len() < res.ndim() {
        target.insert(0, 1);
    }
    while res.ndim() > target.len() {
        res = res.sum_axis(ndarray::Axis(0));
    }
    for i in (0..target.len()).rev() {
        if target[i] == 1 && res.shape()[i] > 1 {
            res = res.sum_axis(ndarray::Axis(i)).insert_axis(ndarray::Axis(i));
        }
    }
    while res.ndim() > target_shape.len() {
        res = res.sum_axis(ndarray::Axis(0));
    }
    res
}

// ---------------------------------------------------------------------------
// Tensor core methods
// ---------------------------------------------------------------------------
impl Tensor {
    pub fn new(data_vec: Vec<f64>, shape: &[usize]) -> Self {
        let data = ArrayD::from_shape_vec(shape, data_vec).expect("Shape mismatch");
        let grad = ArrayD::zeros(data.raw_dim());
        let inner = Arc::new(TensorInner {
            data: RwLock::new(data),
            grad: RwLock::new(grad),
            requires_grad: AtomicBool::new(false),
            grad_fn: Mutex::new(None),
        });
        
        Tensor {
            mode: TensorMode::Eager(inner),
            shape: shape.to_vec(),
        }
    }

    pub fn from_array(data: ArrayD<f64>) -> Self {
        let shape = data.shape().to_vec();
        let grad = ArrayD::zeros(data.raw_dim());
        let inner = Arc::new(TensorInner {
            data: RwLock::new(data),
            grad: RwLock::new(grad),
            requires_grad: AtomicBool::new(false),
            grad_fn: Mutex::new(None),
        });

        Tensor {
            mode: TensorMode::Eager(inner),
            shape,
        }
    }

    pub fn randn(shape: &[usize]) -> Self {
        use ndarray_rand::RandomExt;
        let data = ndarray::ArrayD::random(shape, ndarray_rand::rand_distr::StandardNormal);
        Tensor::from_array(data)
    }

    pub fn scale(&self, s: f64) -> Self {
        self.clone() * Tensor::new(vec![s], &[1])
    }

    /// Internal helper to pull Eager inner if present
    pub fn inner(&self) -> Arc<TensorInner> {
        match &self.mode {
            TensorMode::Eager(inner) => inner.clone(),
            TensorMode::Lazy(_) => panic!("Attempted to access Eager data on a Lazy tensor"),
        }
    }

    pub fn set_requires_grad(&self, flag: bool) {
        if let TensorMode::Eager(inner) = &self.mode {
            inner.requires_grad.store(flag, Ordering::Relaxed);
        }
    }

    pub fn requires_grad(&self) -> bool {
        match &self.mode {
            TensorMode::Eager(inner) => inner.requires_grad.load(Ordering::Relaxed),
            TensorMode::Lazy(_) => false, // Lazy tensors are usually for forward blocks
        }
    }

    pub fn is_lazy(&self) -> bool {
        matches!(self.mode, TensorMode::Lazy(_))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data(&self) -> ArrayD<f64> {
        match &self.mode {
            TensorMode::Eager(inner) => inner.data.read().unwrap().clone(),
            TensorMode::Lazy(_) => panic!("Cannot access data of a Lazy tensor"),
        }
    }

    pub fn grad(&self) -> ArrayD<f64> {
        match &self.mode {
            TensorMode::Eager(inner) => inner.grad.read().unwrap().clone(),
            TensorMode::Lazy(_) => panic!("Cannot access grad of a Lazy tensor"),
        }
    }

    pub fn zero_grad(&self) {
        if let TensorMode::Eager(inner) = &self.mode {
            inner.grad.write().unwrap().fill(0.0);
        }
    }

    /// Internal helper for builder a Lazy result
    pub(crate) fn from_lazy(id: NodeId, shape: Vec<usize>) -> Self {
        Tensor {
            mode: TensorMode::Lazy(id),
            shape,
        }
    }

    /// Gets or creates the NodeId for this tensor in the current lazy context.
    pub fn node_id(&self) -> NodeId {
        use crate::compiler::{context, OpKind};
        match &self.mode {
            TensorMode::Lazy(id) => *id,
            TensorMode::Eager(_) => {
                if context::is_tracing() {
                    context::with_graph(|g| {
                        // Register this eager tensor as an Input node
                        crate::compiler::add_op(g, OpKind::Input, vec![], self.shape.clone())
                    })
                } else {
                    panic!("Attempted to get node_id outside of lazy tracing context");
                }
            }
        }
    }

    pub fn backward(&self) {
        self.backward_impl(false);
    }

    /// Applies a custom differentiable function to a set of input tensors.
    pub fn apply<F: CustomFunction + 'static>(func: F, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut ctx = Context::new();
        let input_data: Vec<_> = inputs.iter().map(|t| t.inner().data.read().unwrap().clone()).collect();
        
        let func_arc = Arc::new(func);
        let output_data = func_arc.forward(&input_data, &mut ctx);
        
        let mut requires_grad = inputs.iter().any(|t| t.requires_grad());
        let mut final_grad_fn = None;

        if grad_enabled() && requires_grad {
            let backward = Arc::new(CustomBackward {
                func: func_arc.clone(),
                ctx: Context { saved_tensors: ctx.saved_tensors.clone() },
                parents: inputs.to_vec(),
            });
            final_grad_fn = Some(backward as Arc<dyn BackwardFn + Send + Sync>);
        } else {
            requires_grad = false;
        }

        output_data.into_iter().map(|data| {
            let res = Tensor::from_array(data);
            res.inner().requires_grad.store(requires_grad, Ordering::Relaxed);
            *res.inner().grad_fn.lock().unwrap() = final_grad_fn.clone();
            res
        }).collect()

    }


    fn backward_impl(&self, retain_graph: bool) {
        if let TensorMode::Eager(inner) = &self.mode {
            inner.grad.write().unwrap().fill(1.0);
        } else {
            panic!("Cannot call backward on a Lazy tensor");
        }

        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        for t in topo.iter().rev() {
            let inner = t.inner();
            let grad_fn_opt = {
                let lock = inner.grad_fn.lock().unwrap();
                lock.clone()
            };

            if let Some(op) = grad_fn_opt {
                let grad_output = inner.grad.read().unwrap().clone();
                let parent_grads = op.backward(grad_output);

                for (parent, grad_contrib) in op.parents().iter().zip(parent_grads) {
                    if parent.requires_grad() {
                        let p_inner = parent.inner();
                        *p_inner.grad.write().unwrap() += &grad_contrib;
                    }
                }

                if !retain_graph {
                    *inner.grad_fn.lock().unwrap() = None;
                }
            }
        }
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
        let id = match &self.mode {
            TensorMode::Eager(inner) => Arc::as_ptr(inner) as usize,
            TensorMode::Lazy(node_id) => *node_id,
        };

        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let parents: Vec<Tensor> = match &self.mode {
            TensorMode::Eager(inner) => {
                let lock = inner.grad_fn.lock().unwrap();
                match lock.as_ref() {
                    Some(op) => op.parents(),
                    None => vec![],
                }
            },
            TensorMode::Lazy(_) => vec![], // Lazy tensors usually have static inputs
        };

        for p in parents {
            p.build_topo(visited, out);
        }
        out.push(self.clone());
    }

    pub(crate) fn make_result(
        data: ArrayD<f64>,
        mut requires_grad: bool,
        mut grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>>,
    ) -> Tensor {
        if !grad_enabled() {
            requires_grad = false;
            grad_fn = None;
        }

        let grad = ArrayD::zeros(data.raw_dim());
        let shape = data.shape().to_vec();
        let inner = Arc::new(TensorInner {
            data: RwLock::new(data),
            grad: RwLock::new(grad),
            requires_grad: AtomicBool::new(requires_grad),
            grad_fn: Mutex::new(grad_fn),
        });

        Tensor {
            mode: TensorMode::Eager(inner),
            shape,
        }
    }
}
