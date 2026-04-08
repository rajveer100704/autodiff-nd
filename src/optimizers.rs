use ndarray::ArrayD;


use crate::engine::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

// ---------------------------------------------------------------------------
// SGD — Stochastic Gradient Descent
// w = w - lr * grad
// ---------------------------------------------------------------------------
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
    fn step(&mut self) {
        for p in &self.params {
            if p.requires_grad() {
                // Read grad, drop the read lock, then write data.
                // INVARIANT: never hold read + write locks simultaneously.
                let grad = p.inner().grad.read().unwrap().clone();
                let inner = p.inner();
                let mut data = inner.data.write().unwrap();
                *data = &*data - &(grad * self.lr);
            }
        }
    }

    fn zero_grad(&mut self) {
        for p in &self.params {
            p.inner().grad.write().unwrap().fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Adam — Adaptive Moment Estimation (with bias correction)
//
// Parameters:
//   lr    — learning rate
//   beta1 — 1st moment decay (momentum)
//   beta2 — 2nd moment decay (uncentered variance)
//   eps   — numerical stability constant
// ---------------------------------------------------------------------------
pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
    m: Vec<ArrayD<f64>>, // 1st moment (momentum)
    v: Vec<ArrayD<f64>>, // 2nd moment (uncentered variance)
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &params {
            let shape = p.inner().data.read().unwrap().raw_dim();
            m.push(ArrayD::zeros(shape.clone()));
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
            p.inner().grad.write().unwrap().fill(0.0);
        }
    }

    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f64;

        for (i, p) in self.params.iter().enumerate() {
            if !p.requires_grad() {
                continue;
            }

            // Read grad — drop lock before writing
            let grad = p.inner().grad.read().unwrap().clone();

            // Update moments
            self.m[i] = &self.m[i] * self.beta1 + &grad * (1.0 - self.beta1);
            self.v[i] = &self.v[i] * self.beta2 + (&grad * &grad) * (1.0 - self.beta2);

            // Bias-corrected estimates
            let m_hat = &self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = &self.v[i] / (1.0 - self.beta2.powf(t));

            // Parameter update: w = w - lr * m_hat / (sqrt(v_hat) + eps)
            let v_hat_sqrt = v_hat.mapv(|x| x.sqrt() + self.eps);
            let inner = p.inner();
            let mut data = inner.data.write().unwrap();
            *data = &*data - &((&m_hat / &v_hat_sqrt) * self.lr);
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient utilities
// ---------------------------------------------------------------------------

/// Clip gradients so that the global L2 norm ≤ max_norm.
/// This prevents exploding gradients in deep networks.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f64) {
    // Compute global norm across all parameters
    let total_norm = params
        .iter()
        .map(|p| {
            p.inner().grad
                .read()
                .unwrap()
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();

    if total_norm > max_norm {
        let clip_coeff = max_norm / (total_norm + 1e-6);
        for p in params {
            p.inner().grad
                .write()
                .unwrap()
                .mapv_inplace(|x| x * clip_coeff);
        }
    }
}

/// L2 regularization penalty: λ * Σ ||W||²
/// Only penalizes weight matrices (ndim > 1), not bias vectors.
pub fn l2_regularization(params: Vec<Tensor>, lambda: f64) -> Tensor {
    let l = Tensor::new(vec![lambda], &[1]);
    let mut penalty = Tensor::new(vec![0.0], &[1]);
    for p in params {
        if p.data().ndim() > 1 {
            let squared_sum = (p.clone() * p.clone()).sum();
            penalty = penalty + (l.clone() * squared_sum);
        }
    }
    penalty
}
