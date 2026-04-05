use ndarray::{ArrayD, Zip};
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
        let g1 = reduce_grad(
            -(&grad_output * &p0.data) / p1.data.mapv(|v| v.powi(2)),
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
        let grad_input = grad_output / (sqrt_x * 2.0);

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

// --- Matmul Implementation ---

struct MatmulBackward {
    parents: Vec<Tensor>,
}

impl BackwardFn for MatmulBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let a = self.parents[0].0.borrow().data.clone();
        let b = self.parents[1].0.borrow().data.clone();

        // Convert ArrayD to Array2 temporarily for 2D matrix multiplication
        let g_out = grad_output.into_dimensionality::<ndarray::Ix2>().unwrap();
        let a_mat = a.into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_mat = b.into_dimensionality::<ndarray::Ix2>().unwrap();

        // dA = grad_output @ B^T

        let grad_a = g_out.dot(&b_mat.t()).into_dyn();

        // dB = A^T @ grad_output
        let grad_b = a_mat.t().dot(&g_out).into_dyn();

        vec![grad_a, grad_b]
    }

    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        let req_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;

        let a_inner = self.0.borrow();
        let b_inner = rhs.0.borrow();

        // Perform matrix multiplication
        // into_dimensionality converts ArrayD to Array2
        let a_mat = a_inner
            .data
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("A must be 2D");
        let b_mat = b_inner
            .data
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("B must be 2D");

        let res_data = a_mat.dot(&b_mat).into_dyn();
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

/* ------------------- Loss Functions ------------------ */

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    // MSE = 1/n sum<i=1;n> (predi - targeti)^2
    let diff = pred.clone() - target.clone();
    let squared = diff.pow(2.0);
    squared.mean()
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Vec<usize>) -> Tensor {
    let exp = logits.exp();
    let sum_exp = exp.sum(); // scalar
    let log_sum_exp = sum_exp.ln();

    let mut mask_vec = vec![0.0; logits.data().len()];
    mask_vec[targets[0]] = 1.0;

    let mask = Tensor::new(mask_vec, logits.data().shape());
    let correct_logit = (logits.clone() * mask).sum();

    log_sum_exp - correct_logit
}

pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let one = Tensor::new(vec![1.0], pred.data().shape());

    //y * ln(p)
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
