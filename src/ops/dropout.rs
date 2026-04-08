use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use ndarray::ArrayD;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Bernoulli;
use crate::engine::{Tensor, BackwardFn};
use crate::module::Module;

pub struct DropoutBackward {
    pub parent: Tensor,
    pub mask: ArrayD<f64>,
    pub scale: f64,
}

impl BackwardFn for DropoutBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output * &self.mask * self.scale]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Dropout"
    }
}

pub struct Dropout {
    pub p: f64,
    pub is_training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Self { 
            p,
            is_training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout {
    fn set_training(&self, flag: bool) {
        self.is_training.store(flag, Ordering::Relaxed);
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.is_training.load(Ordering::Relaxed) {
            return x.clone();
        }
        
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;
        let inner = x.inner();
        let data = inner.data.read().unwrap();
        let shape = data.raw_dim();
        
        let mask = ArrayD::random(shape, Bernoulli::new(keep_prob).unwrap())
            .mapv(|b| if b { 1.0 } else { 0.0 });
            
        let res_data = (&*data * &mask) * scale;
        drop(data);

        let req_grad = x.requires_grad();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(DropoutBackward {
                parent: x.clone(),
                mask,
                scale,
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // Dropout has no learnable parameters
    }
}
