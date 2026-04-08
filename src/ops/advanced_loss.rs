use std::sync::Arc;
use ndarray::{ArrayD, Axis, Zip};
use crate::engine::{Tensor, BackwardFn};

// --- NLLLoss (Negative Log Likelihood) ---
// Expects log-probabilities as input and integer class indices as targets.
// However, in this framework, we'll assume 'target' is a one-hot or probability tensor 
// of the same shape to keep the math generic.
pub struct NllLossBackward {
    pub input: Tensor,
    pub target: Tensor,
}

impl BackwardFn for NllLossBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let target_data = self.target.data();
        let batch_size = target_data.shape()[0] as f64;
        // dL/dinput = -target_data / batch_size * grad_output
        vec![-&target_data / batch_size * grad_output]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.target.clone()]
    }
    fn op_name(&self) -> &'static str {
        "NLLLoss"
    }
}

pub fn nll_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let req_grad = input.requires_grad() || target.requires_grad();
    let x = input.data();
    let y = target.data();
    let _batch_size = x.shape()[0] as f64;

    // Loss = -sum(input * target) / batch_size
    let res_data = -(&x * &y).sum_axis(Axis(x.ndim() - 1)).mean_axis(Axis(0)).unwrap().into_dyn();

    let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
        Some(Arc::new(NllLossBackward {
            input: input.clone(),
            target: target.clone(),
        }))
    } else {
        None
    };
    Tensor::make_result(res_data, req_grad, grad_fn)
}

// --- KLDivLoss (Kullback-Leibler Divergence) ---
// input is log-probabilities, target is probabilities.
pub struct KlDivLossBackward {
    pub input: Tensor,
    pub target: Tensor,
}

impl BackwardFn for KlDivLossBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let target_data = self.target.data();
        let batch_size = target_data.shape()[0] as f64;
        // dL/dinput = -target_data / batch_size
        vec![-&target_data / batch_size * grad_output]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.target.clone()]
    }
    fn op_name(&self) -> &'static str {
        "KLDivLoss"
    }
}

pub fn kl_div_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let req_grad = input.requires_grad() || target.requires_grad();
    let x_data = input.data();
    let y_data = target.data();
    let _batch_size = x_data.shape()[0] as f64;

    // L = target * (log(target) - input). Handling target=0 safely.
    let mut res_data = ArrayD::zeros(x_data.raw_dim());
    Zip::from(&mut res_data).and(&x_data).and(&y_data).for_each(|r, &ix, &iy| {
        if iy > 0.0 {
            *r = iy * (iy.ln() - ix);
        } else {
            *r = 0.0;
        }
    });
    
    let res_final = res_data.sum_axis(Axis(x_data.ndim() - 1)).mean_axis(Axis(0)).unwrap().into_dyn();

    let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
        Some(Arc::new(KlDivLossBackward {
            input: input.clone(),
            target: target.clone(),
        }))
    } else {
        None
    };
    Tensor::make_result(res_final, req_grad, grad_fn)
}

// --- HuberLoss ---
pub struct HuberLossBackward {
    pub input: Tensor,
    pub target: Tensor,
    pub delta: f64,
}

impl BackwardFn for HuberLossBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.input.data();
        let y = self.target.data();
        let batch_size = x.len() as f64;
        let mut grad = ArrayD::zeros(x.raw_dim());
        
        Zip::from(&mut grad).and(&x).and(&y).for_each(|g, &ix, &iy| {
            let diff = ix - iy;
            if diff.abs() <= self.delta {
                *g = diff;
            } else {
                *g = self.delta * diff.signum();
            }
        });
        
        vec![grad / batch_size * grad_output]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.target.clone()]
    }
    fn op_name(&self) -> &'static str {
        "HuberLoss"
    }
}

pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f64) -> Tensor {
    let req_grad = input.requires_grad() || target.requires_grad();
    let x_data = input.data();
    let y_data = target.data();
    let batch_size = x_data.len() as f64;

    let mut loss_sum = 0.0;
    Zip::from(&x_data).and(&y_data).for_each(|&ix, &iy| {
        let diff = (ix - iy).abs();
        if diff <= delta {
            loss_sum += 0.5 * diff * diff;
        } else {
            loss_sum += delta * (diff - 0.5 * delta);
        }
    });

    let res_data = ArrayD::from_elem(vec![], loss_sum / batch_size);

    let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
        Some(Arc::new(HuberLossBackward {
            input: input.clone(),
            target: target.clone(),
            delta,
        }))
    } else {
        None
    };
    Tensor::make_result(res_data, req_grad, grad_fn)
}
