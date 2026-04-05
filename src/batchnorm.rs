use std::cell::{Cell, RefCell};

use ndarray::{ArrayD, Axis, IxDyn};

use crate::{engine::Tensor, module::Module};

/*--------------------------------------------

            BatchNorm1D

----------------------------------------------*/

pub struct BatchNorm1d {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: RefCell<ArrayD<f64>>,
    pub running_var: RefCell<ArrayD<f64>>,
    pub eps: f64,
    pub momentum: f64,
    pub is_training: Cell<bool>,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        let gamma = Tensor::new(vec![1.0; num_features], &[1, num_features]);
        gamma.set_requires_grad(true);

        let beta = Tensor::new(vec![0.0; num_features], &[1, num_features]);
        beta.set_requires_grad(true);

        Self {
            gamma,
            beta,
            running_mean: RefCell::new(ArrayD::zeros(IxDyn(&[1, num_features]))),
            running_var: RefCell::new(ArrayD::ones(IxDyn(&[1, num_features]))),
            eps: 1e-5,
            momentum: 0.1,
            is_training: Cell::new(false),
        }
    }
}

impl Module for BatchNorm1d {
    fn set_training(&self, is_training: bool) {
        self.is_training.set(is_training);
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        if self.is_training.get() {
            // -----------------------------
            // TRAINING MODE
            // -----------------------------

            let x_data = x.data();

            let batch_mean = x_data.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));

            let batch_var = x_data.var_axis(Axis(0), 0.0).insert_axis(Axis(0));

            // --- Update running stats ---
            {
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();

                *rm = &*rm * (1.0 - self.momentum) + &batch_mean * self.momentum;

                *rv = &*rv * (1.0 - self.momentum) + &batch_var * self.momentum;
            }

            // --- Convert to Tensor ---
            let mean_t = Tensor::from_array(batch_mean);
            let var_t = Tensor::from_array(batch_var);

            // eps tensor (broadcast-safe)
            let eps_t = Tensor::from_array(ArrayD::from_elem(var_t.data().raw_dim(), self.eps));

            // std = sqrt(var + eps)
            let std_t = (var_t + eps_t).sqrt();

            // normalize
            let x_hat = (x.clone() - mean_t) / std_t;

            // scale + shift
            (self.gamma.clone() * x_hat) + self.beta.clone()
        } else {
            // -----------------------------
            // INFERENCE MODE
            // -----------------------------

            let mean_t = Tensor::from_array(self.running_mean.borrow().clone());

            let var_t = Tensor::from_array(self.running_var.borrow().clone());

            let eps_t = Tensor::from_array(ArrayD::from_elem(var_t.data().raw_dim(), self.eps));

            let std_t = (var_t + eps_t).sqrt();

            let x_hat = (x.clone() - mean_t) / std_t;

            (self.gamma.clone() * x_hat) + self.beta.clone()
        }
    }
}
