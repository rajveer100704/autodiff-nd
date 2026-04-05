use ndarray::ArrayD;

use crate::engine::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

/*
SGD -Stochastic Gradient Descent.

Updates parameter after evaluating each training example(small batch) rather than
entire dataset
*/

pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f64,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl Optimizer for SGD {
    // w = w - lr * grad
    fn step(&mut self) {
        for p in &self.params {
            let mut inner = p.0.borrow_mut();
            if inner.requires_grad {
                inner.data = &inner.data - &(inner.grad.clone() * self.lr);
            }
        }
    }

    fn zero_grad(&mut self) {
        for p in &self.params {
            let mut inner = p.0.borrow_mut();
            inner.grad.fill(0.0);
        }
    }
}

/*-------------------------------END SGD------------------------------------ */

/*
    Adam Optimizer - Adaptive Moment Estimation

    - has adaptive learning rates
    - bias correction
    - has low memory requirements

    parameters:
    1. learning rate - lr
    2. momentum - beta1
    3. uncentered variance - beta2
    4. constant to prevent division by zero - eps
*/

pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,            // timestamp (counter)
    m: Vec<ArrayD<f64>>, // First momentum (velocity)
    v: Vec<ArrayD<f64>>, // Second momentum (vibration)
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &params {
            let data = p.data();
            let shape = data.shape();
            m.push(ArrayD::zeros(shape));
            v.push(ArrayD::zeros(shape));
        }
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m,
            v,
        }
    }
}

impl Optimizer for Adam {
    fn zero_grad(&mut self) {
        for p in &self.params {
            p.0.borrow_mut().grad.fill(0.0);
        }
    }
    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f64;

        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.0.borrow_mut();

            if !inner.requires_grad {
                continue;
            }

            let grad = &inner.grad;

            // update momentum
            self.m[i] = &self.m[i] * self.beta1 + grad * (1.0 - self.beta1);

            // update variance
            self.v[i] = &self.v[i] * self.beta2 + (grad * grad) * (1.0 - self.beta2);

            // bias correction
            let m_hat = &self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = &self.v[i] / (1.0 - self.beta2.powf(t));

            // update weights
            // w = w - lr * m_hat / (sqrt(v_hat) + eps)

            let v_hat_sqrt = v_hat.mapv(|x| x.sqrt() + self.eps);
            inner.data = &inner.data - &((&m_hat / &v_hat_sqrt) * self.lr);
        }
    }
}

/*-------------------------------END ADAM------------------------------------ */

pub fn clip_grad_norm(params: Vec<Tensor>, max_norm: f64) {
    // 1. Calculate total L2 norm
    let mut total_norm = 0.0;
    for p in &params {
        let inner = p.0.borrow();
        // Sum of squares of all elements in this tensor's gradient
        total_norm += inner.grad.iter().map(|&x| x * x).sum::<f64>();
    }
    total_norm = total_norm.sqrt();

    // 2. If norm exceeds max, scale everything down
    if total_norm > max_norm {
        let clip_coeff = max_norm / (total_norm + 1e-6); // 1e-6 for numerical stability
        for p in &params {
            let mut inner = p.0.borrow_mut();
            inner.grad.mapv_inplace(|x| x * clip_coeff);
        }
    }
}

pub fn l2_regularization(params: Vec<Tensor>, lambda: f64) -> Tensor {
    // Create a constant tensor for the multiplier
    let l = Tensor::new(vec![lambda], &[1]);
    let mut penalty = Tensor::new(vec![0.0], &[1]);

    for p in params {
        // Only penalize weights (rank > 1), skip biases (rank 1)
        if p.data().ndim() > 1 {
            // p * p (element-wise) -> sum() (scalar tensor)
            let squared_sum = (p.clone() * p.clone()).sum();

            // accumulation: penalty = penalty + (l * squared_sum)
            penalty = penalty + (l.clone() * squared_sum);
        }
    }
    penalty
}
