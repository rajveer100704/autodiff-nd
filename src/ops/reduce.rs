use std::sync::Arc;

use ndarray::ArrayD;
use crate::engine::{Tensor, BackwardFn};

// --- Sum (all elements → scalar) ---
pub struct SumBackward {
    pub parent: Tensor,
    pub input_shape: Vec<usize>,
}

impl BackwardFn for SumBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_val = grad_output[[0]];
        vec![ArrayD::from_elem(self.input_shape.clone(), grad_val)]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Sum"
    }
}

impl Tensor {
    pub fn sum(&self) -> Tensor {
        let req_grad = self.requires_grad();
        let inner = self.inner();
        let data = inner.data.read().unwrap();
        let input_shape = data.shape().to_vec();
        let sum_val = data.sum();
        drop(data);
        let res_data = ArrayD::from_elem(vec![1], sum_val);
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SumBackward {
                parent: self.clone(),
                input_shape,
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Mean ---
pub struct MeanBackward {
    pub parent: Tensor,
    pub input_shape: Vec<usize>,
    pub n: f64,
}

impl BackwardFn for MeanBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_val = grad_output[[0]] / self.n;
        vec![ArrayD::from_elem(self.input_shape.clone(), grad_val)]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Mean"
    }
}

impl Tensor {
    pub fn mean(&self) -> Tensor {
        let sum = self.sum();
        let n = Tensor::new(vec![self.numel() as f64], &[1]);
        sum / n
    }
}

// --- SumAxis (reduce along one axis, keepdim=true) ---
pub struct SumAxisBackward {
    pub parent: Tensor,
    pub axis: usize,
    pub input_shape: Vec<usize>,
}

impl BackwardFn for SumAxisBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let mut grad = grad_output;
        if grad.ndim() < self.input_shape.len() {
            grad = grad.insert_axis(ndarray::Axis(self.axis));
        }
        let grad_input = grad.broadcast(self.input_shape.clone()).unwrap().to_owned();
        vec![grad_input]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "SumAxis"
    }
}

impl Tensor {
    pub fn sum_axis(&self, axis: usize) -> Self {
        let req_grad = self.requires_grad();
        let inner = self.inner();
        let data = inner.data.read().unwrap();
        let input_shape = data.shape().to_vec();
        let reduced = data
            .sum_axis(ndarray::Axis(axis))
            .insert_axis(ndarray::Axis(axis));
        drop(data);
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SumAxisBackward {
                parent: self.clone(),
                axis,
                input_shape,
            }))
        } else {
            None
        };
        Tensor::make_result(reduced, req_grad, grad_fn)
    }
}
