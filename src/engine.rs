use ndarray::ArrayD;
use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

// The Inner State: Shared across all Tensor clones
pub struct TensorInner {
    pub data: ArrayD<f64>,
    pub grad: ArrayD<f64>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<dyn BackwardFn>>,
}

// The Public Handle: A simple wrapper around Rc<RefCell<...>>
#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

pub trait BackwardFn {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>>;
    fn parents(&self) -> Vec<Tensor>;
}

impl Tensor {
    pub fn new(data_vec: Vec<f64>, shape: &[usize]) -> Self {
        let data = ArrayD::from_shape_vec(shape, data_vec).expect("Shape mismatch");
        let grad = ArrayD::zeros(shape);

        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad,
            requires_grad: false,
            grad_fn: None,
        })))
    }

    pub fn set_requires_grad(&self, flag: bool) {
        self.0.borrow_mut().requires_grad = flag;
    }

    pub fn backward(&self) {
        // Initialize the seed gradient (dy/dy = 1.0)
        {
            let mut inner = self.0.borrow_mut();
            inner.grad.fill(1.0);
        }

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        // Process in reverse topological order (root to leaves)
        for t in topo.iter().rev() {
            let mut t_inner = t.0.borrow_mut();

            if let Some(op) = t_inner.grad_fn.take() {
                // .take() breaks the cycle during traversal
                let grad_output = t_inner.grad.clone();

                // Drop borrow before calling parents to avoid RefCell panics
                drop(t_inner);

                let parent_grads = op.backward(grad_output);
                for (parent, grad_contrib) in op.parents().iter().zip(parent_grads) {
                    let mut p_inner = parent.0.borrow_mut();
                    if p_inner.requires_grad {
                        p_inner.grad += &grad_contrib;
                    }
                }
            }
        }
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
        let id = Rc::as_ptr(&self.0) as usize;
        if !visited.contains(&id) {
            visited.insert(id);
            let inner = self.0.borrow();
            if let Some(op) = &inner.grad_fn {
                for p in op.parents() {
                    p.build_topo(visited, out);
                }
            }
            out.push(self.clone());
        }
    }

    pub fn data(&self) -> ArrayD<f64> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> ArrayD<f64> {
        self.0.borrow().grad.clone()
    }
}

// --- Add Implementation ---

struct AddBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for AddBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Derivative of x + y is 1, so pass grad through
        vec![grad_output.clone(), grad_output]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;
        let res_data = &self.0.borrow().data + &rhs.0.borrow().data;
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(AddBackward {
                parents: vec![self, rhs],
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Mul Implementation ---

struct MulBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for MulBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let p0 = self.parents[0].0.borrow();
        let p1 = self.parents[1].0.borrow();
        // d(x*y)/dx = y, d(x*y)/dy = x
        vec![&grad_output * &p1.data, &grad_output * &p0.data]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;
        let res_data = &self.0.borrow().data * &rhs.0.borrow().data;
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(MulBackward {
                parents: vec![self, rhs],
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Sub Implementation ---

struct SubBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for SubBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_y = grad_output.clone().neg();
        vec![grad_output.clone(), grad_y]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;
        let res_data = &self.0.borrow().data - &rhs.0.borrow().data;
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SubBackward {
                parents: vec![self, rhs],
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Neg Implementation ---

struct NegBackward {
    parent: Tensor,
}

impl BackwardFn for NegBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![-grad_output]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let req_grad = self.0.borrow().requires_grad;
        let res_data = -self.0.borrow().data.clone();
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(NegBackward { parent: self }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Pow Implementation ---

struct PowBackward {
    parent: Tensor,
    exponent: f64,
}

impl BackwardFn for PowBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let n = self.exponent;
        // Short-lived borrow of the parent's data
        let grad_x = {
            let p_inner = self.parent.0.borrow();
            grad_output * n * p_inner.data.mapv(|v| v.powf(n - 1.0))
        };
        vec![grad_x]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn pow(self, n: f64) -> Self {
        let req_grad = self.0.borrow().requires_grad;
        let res_data = self.0.borrow().data.mapv(|v| v.powf(n));
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(PowBackward {
                parent: self,
                exponent: n,
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Sum() Implementation ---

struct SumBackward {
    parent: Tensor,
    input_shape: Vec<usize>,
}

impl BackwardFn for SumBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // grad_output is a 1-element array (the gradient of the scalar sum).
        // We broadcast that value to the entire shape of the input.
        let grad_val = grad_output[[0]];
        let grad_input = ArrayD::from_elem(self.input_shape.clone(), grad_val);
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn sum(&self) -> Tensor {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        // Compute scalar sum and convert to 1x1 ArrayD
        let sum_val = inner.data.sum();
        let res_data = ArrayD::from_elem(vec![1], sum_val);
        let res_grad = ArrayD::zeros(vec![1]);
        let input_shape = inner.data.shape().to_vec();

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SumBackward {
                parent: self.clone(),
                input_shape,
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Exp Implementation ---

struct ExpBackward {
    parent: Tensor,
}

impl BackwardFn for ExpBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Derivative of e^x is e^x.
        let x = self.parent.0.borrow().data.clone();
        let exp_x = x.mapv(|v| v.exp());
        vec![grad_output * exp_x]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn exp(&self) -> Tensor {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        let res_data = inner.data.mapv(|v| v.exp());
        let res_grad = ArrayD::zeros(vec![1]);

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(ExpBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Mean Implementation ---

struct MeanBackward {
    parent: Tensor,
    input_shape: Vec<usize>,
    n: f64, // Total number of elements
}

impl BackwardFn for MeanBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // grad_output is the gradient of the scalar mean.
        // We broadcast (grad_output / n) to the entire original shape.
        let grad_val = grad_output[[0]] / self.n;
        let grad_input = ArrayD::from_elem(self.input_shape.clone(), grad_val);
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}
impl Tensor {
    pub fn mean(&self) -> Tensor {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        let n = inner.data.len() as f64;
        let mean_val = inner
            .data
            .mean()
            .expect("Cannot compute mean of empty tensor");

        let res_data = ArrayD::from_elem(vec![1], mean_val);
        let res_grad = ArrayD::zeros(vec![1]);
        let input_shape = inner.data.shape().to_vec();

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(MeanBackward {
                parent: self.clone(),
                input_shape,
                n,
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- ln() Implementation ---

struct LnBackward {
    parent: Tensor,
}

impl BackwardFn for LnBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // d(ln x)/dx = 1/x
        let x = self.parent.0.borrow().data.clone();
        // grad_output * (1 / x)
        let grad_input = grad_output / x;
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn ln(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        // Compute ln(v) for every element v in data
        let res_data = inner.data.mapv(|v| v.ln());
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(LnBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

/* --------------------------------------
        Activation Functions
----------------------------------------- */

// --- ReLu Implementation ---

struct ReluBackward {
    parent: Tensor,
}

impl BackwardFn for ReluBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.0.borrow().data.clone();

        let mask = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        let grad_input = grad_output * mask;

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn relu(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        // Apply max(0, v) to every element
        let res_data = inner.data.mapv(|v| v.max(0.0));
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(ReluBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: res_data,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}
