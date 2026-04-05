use ndarray::{Array2, ArrayD, Zip};
use ndarray_rand::{RandomExt, rand_distr::Bernoulli};
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

/* --------------------- global functions-------------------------- */
fn reduce_grad(grad: ArrayD<f64>, target_shape: &[usize]) -> ArrayD<f64> {
    let mut res = grad;

    // Step 1: Right-align target shape (pad with 1s on the left)
    let mut target = target_shape.to_vec();
    while target.len() < res.ndim() {
        target.insert(0, 1);
    }

    // Step 2: Reduce extra dimensions
    while res.ndim() > target.len() {
        res = res.sum_axis(ndarray::Axis(0));
    }

    // Step 3: Reduce broadcasted dimensions
    for i in (0..target.len()).rev() {
        if target[i] == 1 && res.shape()[i] > 1 {
            res = res.sum_axis(ndarray::Axis(i)).insert_axis(ndarray::Axis(i));
        }
    }

    // Step 4: Remove leading dims if original target was smaller
    while res.ndim() > target_shape.len() {
        res = res.sum_axis(ndarray::Axis(0));
    }

    res
}

/* ------------------------------------------------------------------- */
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

    pub fn from_array(data: ArrayD<f64>) -> Self {
        let grad = ArrayD::zeros(data.raw_dim());

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
        let shape0 = self.parents[0].0.borrow().data.shape().to_vec();
        let shape1 = self.parents[1].0.borrow().data.shape().to_vec();

        vec![
            reduce_grad(grad_output.clone(), &shape0),
            reduce_grad(grad_output, &shape1),
        ]
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

        let shape0 = p0.data.shape().to_vec();
        let shape1 = p1.data.shape().to_vec();

        // d(x*y)/dx = y * grad_output
        let g0 = reduce_grad(&grad_output * &p1.data, &shape0);
        // d(x*y)/dy = x * grad_output
        let g1 = reduce_grad(&grad_output * &p0.data, &shape1);

        vec![g0, g1]
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
        let shape0 = self.parents[0].0.borrow().data.shape().to_vec();
        let shape1 = self.parents[1].0.borrow().data.shape().to_vec();

        let grad_0 = reduce_grad(grad_output.clone(), &shape0);
        let grad_1 = reduce_grad(-grad_output, &shape1);

        vec![grad_0, grad_1]
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

// --- Div Implementation ---

struct DivBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for DivBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let p0 = self.parents[0].0.borrow(); // x
        let p1 = self.parents[1].0.borrow(); // y

        let shape0 = p0.data.shape().to_vec();
        let shape1 = p1.data.shape().to_vec();

        // dx = grad_output * (1 / y)
        let g0 = reduce_grad(&grad_output / &p1.data, &shape0);

        // dy = grad_output * (-x / y^2)
        let eps = 1e-12;

        let g1 = reduce_grad(
            -(&grad_output * &p0.data) / p1.data.mapv(|v| v * v + eps),
            &shape1,
        );

        vec![g0, g1]
    }

    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

use std::ops::Div;

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;

        let res_data = &self.0.borrow().data / &rhs.0.borrow().data;
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(DivBackward {
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

// --- Sqrt Implementation ---

struct SqrtBackward {
    parent: Tensor,
}

impl BackwardFn for SqrtBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.0.borrow().data.clone();

        // sqrt(x)
        let sqrt_x = x.mapv(|v| v.sqrt());

        // grad = grad_output * (1 / (2 * sqrt(x)))
        let eps = 1e-12;
        let grad_input = grad_output / ((sqrt_x + eps) * 2.0);

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn sqrt(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        let res_data = inner.data.mapv(|v| v.sqrt());
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SqrtBackward {
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
        let res_grad = ArrayD::zeros(res_data.raw_dim());

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
        let eps = 1e-12;
        let res_data = inner.data.mapv(|v| (v.max(eps)).ln());
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

// --- Sigmoid Implementation ---

struct SigmoidBackward {
    parent: Tensor,
}

impl BackwardFn for SigmoidBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.0.borrow().data.clone();

        // calculate sigmoid
        let s = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));

        // grad = grad_output * s * (1 - s)
        let grad_input = grad_output * &s * (ArrayD::from_elem(s.raw_dim(), 1.0) - &s);

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn sigmoid(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        // Apply max(0, v) to every element
        let res_data = inner.data.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SigmoidBackward {
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

// --- Tanh Implementation ---

struct TanhBackward {
    parent: Tensor,
}

impl BackwardFn for TanhBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.0.borrow().data.clone();

        // calculate tanh
        let t = x.mapv(|v| v.tanh());

        // grad: grad_output * (1 - t^2)
        let grad_input = grad_output * (ArrayD::from_elem(t.raw_dim(), 1.0) - (&t * &t));

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn tanh(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        // Apply max(0, v) to every element
        let res_data = inner.data.mapv(|v| v.tanh());
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(TanhBackward {
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

// --- Gelu Implementation (Modern variation used in trandformers) ---
// Gaussian Error Linear Unit

struct GeluBackward {
    parent: Tensor,
}

impl BackwardFn for GeluBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.0.borrow().data.clone();

        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();

        let mut grad_input = grad_output.clone();

        Zip::from(&mut grad_input).and(&x).for_each(|g, &v| {
            let x3 = v * v * v;
            let inner = sqrt_2_pi * (v + 0.044715 * x3);
            let tanh_inner = inner.tanh();

            // Derivative components
            let left = 0.5 * (1.0 + tanh_inner);
            let right = 0.5
                * v
                * (1.0 - tanh_inner * tanh_inner)
                * sqrt_2_pi
                * (1.0 + 3.0 * 0.044715 * v * v);

            *g *= left + right;
        });

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn gelu(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;

        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
        let res_data = inner
            .data
            .mapv(|x| 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x * x * x)).tanh()));

        let res_grad = ArrayD::zeros(res_data.raw_dim());
        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(GeluBackward {
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

/* ------------Matrix operations-------------- */

// --- Matmul Implementation (2D and batched 3D)---

fn matmul_2d(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

struct MatmulBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for MatmulBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let a = self.parents[0].0.borrow().data.clone();
        let b = self.parents[1].0.borrow().data.clone();

        let ndim = grad_output.ndim();

        if ndim == 2 {
            // Standard 2D: dA = G @ B^T,  dB = A^T @ G
            let g = grad_output.into_dimensionality::<ndarray::Ix2>().unwrap();
            let a2 = a.into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2 = b.into_dimensionality::<ndarray::Ix2>().unwrap();

            let grad_a = g.dot(&b2.t()).into_dyn();
            let grad_b = a2.t().dot(&g).into_dyn();
            vec![grad_a, grad_b]
        } else if ndim == 3 {
            // Batched 3D: (batch, M, K) @ (K, N) or (batch, K, N)
            let g3 = grad_output.into_dimensionality::<ndarray::Ix3>().unwrap();
            let batch = g3.shape()[0];

            let b_is_2d = b.ndim() == 2;

            let mut grad_a_out = ndarray::Array3::<f64>::zeros((batch, a.shape()[1], a.shape()[2]));
            let mut grad_b_out = ArrayD::zeros(b.raw_dim());

            for i in 0..batch {
                let gi = g3.index_axis(ndarray::Axis(0), i); // (M, N)
                let ai = a
                    .index_axis(ndarray::Axis(0), i)
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap(); // (M, K)

                if b_is_2d {
                    let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    // dA[i] = G[i] @ B^T
                    grad_a_out
                        .index_axis_mut(ndarray::Axis(0), i)
                        .assign(&gi.dot(&b2.t()));
                    // dB += A[i]^T @ G[i]
                    let db = ai
                        .t()
                        .dot(&gi.into_dimensionality::<ndarray::Ix2>().unwrap());
                    grad_b_out += &db.into_dyn();
                } else {
                    let bi = b
                        .index_axis(ndarray::Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap(); // (K, N)
                    let gi2 = gi.into_dimensionality::<ndarray::Ix2>().unwrap();

                    grad_a_out
                        .index_axis_mut(ndarray::Axis(0), i)
                        .assign(&gi2.dot(&bi.t()));

                    let mut gb3 = grad_b_out
                        .view_mut()
                        .into_dimensionality::<ndarray::Ix3>()
                        .unwrap();
                    gb3.index_axis_mut(ndarray::Axis(0), i)
                        .assign(&ai.t().dot(&gi2));
                }
            }

            vec![grad_a_out.into_dyn(), grad_b_out]
        } else {
            panic!("matmul only supports 2D and 3D tensors, got {}D", ndim);
        }
    }

    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;

        let a = self.0.borrow().data.clone();
        let b = rhs.0.borrow().data.clone();

        let res_data = match (a.ndim(), b.ndim()) {
            (2, 2) => {
                let a2 = a.into_dimensionality::<ndarray::Ix2>().unwrap();
                let b2 = b.into_dimensionality::<ndarray::Ix2>().unwrap();
                a2.dot(&b2).into_dyn()
            }
            (3, 2) => {
                // (batch, M, K) @ (K, N) → (batch, M, N)
                let a3 = a.into_dimensionality::<ndarray::Ix3>().unwrap();
                let b2 = b.into_dimensionality::<ndarray::Ix2>().unwrap();
                let batch = a3.shape()[0];
                let m = a3.shape()[1];
                let n = b2.shape()[1];
                let mut out = ndarray::Array3::<f64>::zeros((batch, m, n));
                for i in 0..batch {
                    let ai = a3.index_axis(ndarray::Axis(0), i);
                    let ai2 = ai.into_dimensionality::<ndarray::Ix2>().unwrap();
                    out.index_axis_mut(ndarray::Axis(0), i)
                        .assign(&ai2.dot(&b2));
                }
                out.into_dyn()
            }
            (3, 3) => {
                // (batch, M, K) @ (batch, K, N) → (batch, M, N)
                let a3 = a.into_dimensionality::<ndarray::Ix3>().unwrap();
                let b3 = b.into_dimensionality::<ndarray::Ix3>().unwrap();
                let batch = a3.shape()[0];
                let m = a3.shape()[1];
                let n = b3.shape()[2];
                let mut out = ndarray::Array3::<f64>::zeros((batch, m, n));
                for i in 0..batch {
                    let ai = a3
                        .index_axis(ndarray::Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let bi = b3
                        .index_axis(ndarray::Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    out.index_axis_mut(ndarray::Axis(0), i).assign(&ai.dot(&bi));
                }
                out.into_dyn()
            }
            (a_ndim, b_ndim) => panic!("matmul: unsupported shapes {}D @ {}D", a_ndim, b_ndim),
        };

        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(MatmulBackward {
                parents: vec![self.clone(), rhs.clone()],
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

// --- Transpose Implementation ---

struct TransposeBackward {
    parent: Tensor,
}

impl BackwardFn for TransposeBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Transpose the gradient back
        let ndim = grad_output.ndim();
        let transposed = if ndim == 2 {
            let g2 = grad_output.into_dimensionality::<ndarray::Ix2>().unwrap();
            g2.t().to_owned().into_dyn()
        } else if ndim == 3 {
            // Swap last two axes: (batch, N, M) → (batch, M, N)
            grad_output
                .permuted_axes(ndarray::IxDyn(&[0, 2, 1]))
                .as_standard_layout()
                .to_owned()
                .into_dyn()
        } else {
            panic!("transpose only supports 2D and 3D tensors");
        };
        vec![transposed]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    /// Transposes the last two dimensions.
    /// 2D: (M, N) → (N, M)
    /// 3D: (batch, M, N) → (batch, N, M)
    pub fn transpose(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;
        let ndim = inner.data.ndim();

        let res_data = if ndim == 2 {
            let d2 = inner
                .data
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            d2.t().to_owned().into_dyn()
        } else if ndim == 3 {
            inner
                .data
                .clone()
                .permuted_axes(ndarray::IxDyn(&[0, 2, 1]))
                .as_standard_layout()
                .to_owned()
                .into_dyn()
        } else {
            panic!("transpose only supports 2D and 3D tensors");
        };

        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(TransposeBackward {
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

/* ------------------- Loss Functions ------------------ */

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    // MSE = 1/n sum<i=1;n> (predi - targeti)^2
    let diff = pred.clone() - target.clone();
    let squared = diff.pow(2.0);
    squared.mean()
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let batch = targets.len();
    let num_classes = logits.data().shape()[1];

    // log_softmax = ln(softmax(logits))
    let log_probs = logits.softmax().ln();

    let mut losses = Vec::new();

    for (i, &class) in targets.iter().enumerate() {
        // Build one-hot mask
        let mut mask = vec![0.0; num_classes];
        mask[class] = 1.0;

        let mask_t = Tensor::new(mask, &[num_classes]);

        // Extract row i (we still need slicing workaround)
        let row = logits.slice_row(i);

        let log_row = row.softmax().ln();
        let correct_log_prob = (log_row * mask_t).sum();

        losses.push(-correct_log_prob);
    }

    let mut total = losses[0].clone();
    for l in losses.iter().skip(1) {
        total = total + l.clone();
    }

    let n = Tensor::new(vec![batch as f64], &[1]);
    total / n
}

pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    // Use ones_like instead of hardcoded vec![1.0]
    let one = Tensor::from_array(ArrayD::ones(pred.data().raw_dim()));

    // y * ln(p)
    let term1 = target.clone() * pred.ln();

    // (1 - y) * ln(1 - p)
    let term2 = (one.clone() - target.clone()) * (one - pred.clone()).ln();

    -(term1 + term2).mean()
}

// --- Dropout Implementation ---

struct DropoutBackward {
    parent: Tensor,
    mask: ArrayD<f64>,
    scale: f64,
}

impl BackwardFn for DropoutBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Gradient only flows through the elements that weren't dropped
        // derivative = mask * scale
        let grad_input = grad_output * &self.mask * self.scale;
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

pub struct Dropout {
    pub p: f64, // Probability of dropping an element
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Self { p }
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        if !training {
            return x.clone();
        }

        let inner = x.0.borrow();
        let shape = inner.data.raw_dim();

        // 1. Create a mask where 'keep' probability is (1 - p)
        // We use (1.0 - p) because Bernoulli(p) usually returns 1 with prob p.
        // We want to keep with prob (1-p).
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        let mask = ArrayD::random(shape, Bernoulli::new(keep_prob).unwrap())
            .mapv(|b| if b { 1.0 } else { 0.0 });

        // 2. Apply mask and scale (Inverted Dropout)
        let res_data = (&inner.data * &mask) * scale;
        let res_grad = ArrayD::zeros(res_data.raw_dim());
        let req_grad = inner.requires_grad;

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(DropoutBackward {
                parent: x.clone(),
                mask,
                scale,
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
// --- Softmax Implementation (numerically stable, along last axis) ---

struct SoftmaxBackward {
    input: Tensor, // store input only — avoids Rc cycle
}

impl BackwardFn for SoftmaxBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Recompute softmax(input) from scratch — cheap and cycle-free
        let x = self.input.0.borrow().data.clone();
        let ndim = x.ndim();

        // Numerically stable: subtract max before exp
        let max = x
            .map_axis(ndarray::Axis(ndim - 1), |row| {
                row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            })
            .insert_axis(ndarray::Axis(ndim - 1));

        let exp = (&x - &max).mapv(|v| v.exp());
        let sum = exp
            .sum_axis(ndarray::Axis(ndim - 1))
            .insert_axis(ndarray::Axis(ndim - 1));

        let s = exp / sum; // softmax(x) — shape matches input

        // Jacobian-vector product for softmax:
        // dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
        let dot = (&grad_output * &s)
            .sum_axis(ndarray::Axis(ndim - 1))
            .insert_axis(ndarray::Axis(ndim - 1));

        let grad_input = &s * (&grad_output - &dot);

        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

impl Tensor {
    /// Numerically stable softmax along the last axis.
    /// Works for 1D, 2D (batch, classes), and 3D (batch, seq, features).
    pub fn softmax(&self) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;
        let ndim = inner.data.ndim();

        // Subtract max for numerical stability
        let max = inner
            .data
            .map_axis(ndarray::Axis(ndim - 1), |row| {
                row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            })
            .insert_axis(ndarray::Axis(ndim - 1));

        let shifted = &inner.data - &max;
        let exp = shifted.mapv(|v| v.exp());
        let sum = exp
            .sum_axis(ndarray::Axis(ndim - 1))
            .insert_axis(ndarray::Axis(ndim - 1));

        let res_data = exp / sum;
        let res_grad = ArrayD::zeros(res_data.raw_dim());

        // grad_fn points to self (the input), not the output — no cycle
        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SoftmaxBackward {
                input: self.clone(),
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
// --- sum_axis: reduce along a specific axis, keepdims=true ---

struct SumAxisBackward {
    parent: Tensor,
    axis: usize,
    input_shape: Vec<usize>,
}

impl BackwardFn for SumAxisBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // grad_output has the reduced shape (with the axis kept as size 1).
        // Broadcast it back to the full input shape.
        let mut grad = grad_output;
        // If keepdim collapsed it, re-insert the axis
        if grad.ndim() < self.input_shape.len() {
            grad = grad.insert_axis(ndarray::Axis(self.axis));
        }
        // Broadcast to input shape
        let grad_input = grad.broadcast(self.input_shape.clone()).unwrap().to_owned();
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    /// Sum along a single axis, keeping the dimension (keepdims=true).
    pub fn sum_axis(&self, axis: usize) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;
        let input_shape = inner.data.shape().to_vec();

        let reduced = inner
            .data
            .sum_axis(ndarray::Axis(axis))
            .insert_axis(ndarray::Axis(axis));

        let res_grad = ArrayD::zeros(reduced.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SumAxisBackward {
                parent: self.clone(),
                axis,
                input_shape,
            }))
        } else {
            None
        };

        Tensor(Rc::new(RefCell::new(TensorInner {
            data: reduced,
            grad: res_grad,
            requires_grad: req_grad,
            grad_fn,
        })))
    }
}

// --- Reshape Implementation ---

struct ReshapeBackward {
    parent: Tensor,
    input_shape: Vec<usize>,
}

impl BackwardFn for ReshapeBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Reshape the gradient back to the original input shape
        let grad_input = grad_output
            .into_shape_with_order(self.input_shape.clone())
            .unwrap();
        vec![grad_input]
    }

    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;
        let input_shape = inner.data.shape().to_vec();

        let res_data = inner
            .data
            .clone()
            .into_shape_with_order(new_shape)
            .expect("reshape: total number of elements must match");

        let res_grad = ArrayD::zeros(res_data.raw_dim());

        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(ReshapeBackward {
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

struct SliceRowBackward {
    parent: Tensor,
    row_idx: usize,
}

impl BackwardFn for SliceRowBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let parent_shape = self.parent.0.borrow().data.shape().to_vec();
        let mut grad = ArrayD::zeros(parent_shape);
        grad.index_axis_mut(ndarray::Axis(0), self.row_idx)
            .assign(&grad_output);
        vec![grad]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
}

impl Tensor {
    pub fn slice_row(&self, i: usize) -> Self {
        let inner = self.0.borrow();
        let req_grad = inner.requires_grad;
        let res_data = inner
            .data
            .index_axis(ndarray::Axis(0), i)
            .to_owned()
            .into_dyn();
        let res_grad = ArrayD::zeros(res_data.raw_dim());
        let grad_fn: Option<Rc<dyn BackwardFn>> = if req_grad {
            Some(Rc::new(SliceRowBackward {
                parent: self.clone(),
                row_idx: i,
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
