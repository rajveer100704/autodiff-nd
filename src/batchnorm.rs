use std::sync::atomic::{AtomicBool, Ordering};
use ndarray::{ArrayD, Axis};
use crate::{engine::Tensor, module::Module};

pub struct BatchNorm1d {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
    pub momentum: f64,
    pub is_training: AtomicBool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        let gamma = Tensor::new(vec![1.0; num_features], &[1, num_features]);
        gamma.set_requires_grad(true);

        let beta = Tensor::new(vec![0.0; num_features], &[1, num_features]);
        beta.set_requires_grad(true);

        let running_mean = Tensor::new(vec![0.0; num_features], &[1, num_features]);
        running_mean.set_requires_grad(false);

        let running_var = Tensor::new(vec![1.0; num_features], &[1, num_features]);
        running_var.set_requires_grad(false);

        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            eps: 1e-5,
            momentum: 0.1,
            is_training: AtomicBool::new(false),
        }
    }
}

impl Module for BatchNorm1d {
    fn set_training(&self, flag: bool) {
        self.is_training.store(flag, Ordering::Relaxed);
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
            ("running_mean".to_string(), self.running_mean.clone()),
            ("running_var".to_string(), self.running_var.clone()),
        ]
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        if self.is_training.load(Ordering::Relaxed) {
            let x_data = x.data();
            let batch_mean = x_data.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
            let batch_var = x_data.var_axis(Axis(0), 0.0).insert_axis(Axis(0));

            // Update running stats (momentum)
            {
                let inner_mean = self.running_mean.inner();
                let mut rm = inner_mean.data.write().unwrap();
                *rm = &*rm * (1.0 - self.momentum) + &batch_mean * self.momentum;
            }
            {
                let inner_var = self.running_var.inner();
                let mut rv = inner_var.data.write().unwrap();
                *rv = &*rv * (1.0 - self.momentum) + &batch_var * self.momentum;
            }

            let mean_t = Tensor::from_array(batch_mean);
            let var_t = Tensor::from_array(batch_var);
            let eps_t = Tensor::from_array(ArrayD::from_elem(var_t.data().raw_dim(), self.eps));

            let std_t = (var_t + eps_t).sqrt();
            let x_hat = (x.clone() - mean_t) / std_t;

            self.gamma.clone() * x_hat + self.beta.clone()
        } else {
            let eps_t = Tensor::from_array(ArrayD::from_elem(self.running_var.data().raw_dim(), self.eps));
            let std_t = (self.running_var.clone() + eps_t).sqrt();
            let x_hat = (x.clone() - self.running_mean.clone()) / std_t;

            self.gamma.clone() * x_hat + self.beta.clone()
        }
    }
}
