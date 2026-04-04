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
