use std::sync::Arc;

use ndarray::{ArrayD, Zip};
use crate::engine::{Tensor, BackwardFn};
use crate::compiler::{context, OpKind};

// --- ReLU ---
pub struct ReluBackward {
    pub parent: Tensor,
}

impl BackwardFn for ReluBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.data();
        let mask = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        vec![grad_output * mask]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "ReLU"
    }
}

impl Tensor {
    pub fn relu(&self) -> Tensor {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::ReLU, vec![self.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad();
        let data = self.data();
        let res_data = data.mapv(|x| x.max(0.0));
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(ReluBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Sigmoid ---
pub struct SigmoidBackward {
    pub parent: Tensor,
}

impl BackwardFn for SigmoidBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.data();
        let s = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let ones = ArrayD::from_elem(s.raw_dim(), 1.0);
        vec![grad_output * &s * (ones - &s)]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Sigmoid"
    }
}

impl Tensor {
    pub fn sigmoid(&self) -> Self {
        let req_grad = self.requires_grad();
        let res_data = self.data().mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SigmoidBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Tanh ---
pub struct TanhBackward {
    pub parent: Tensor,
}

impl BackwardFn for TanhBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.inner().data.read().unwrap().clone();
        let t = x.mapv(|v| v.tanh());
        let ones = ArrayD::from_elem(t.raw_dim(), 1.0);
        vec![grad_output * (ones - &t * &t)]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Tanh"
    }
}

impl Tensor {
    pub fn tanh(&self) -> Self {
        let req_grad = self.requires_grad();
        let res_data = self.inner().data.read().unwrap().mapv(|v| v.tanh());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(TanhBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- GELU ---
pub struct GeluBackward {
    pub parent: Tensor,
}

impl BackwardFn for GeluBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.inner().data.read().unwrap().clone();
        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
        let mut grad_input = grad_output.clone();
        Zip::from(&mut grad_input).and(&x).for_each(|g, &v| {
            let x3 = v * v * v;
            let inner = sqrt_2_pi * (v + 0.044715 * x3);
            let tanh_inner = inner.tanh();
            let left = 0.5 * (1.0 + tanh_inner);
            let right = 0.5 * v * (1.0 - tanh_inner * tanh_inner) * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * v * v);
            *g *= left + right;
        });
        vec![grad_input]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "GELU"
    }
}

impl Tensor {
    pub fn gelu(&self) -> Self {
        let req_grad = self.requires_grad();
        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
        let res_data = self.inner().data.read().unwrap().mapv(|x| 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x * x * x)).tanh()));
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(GeluBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Softmax ---
pub struct SoftmaxBackward {
    pub input: Tensor,
}

impl BackwardFn for SoftmaxBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.input.data();
        let ndim = x.ndim();
        let s = softmax_forward(&x);
        let dot = (&grad_output * &s).sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
        vec![&s * (&grad_output - &dot)]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Softmax"
    }
}

pub fn softmax_forward(data: &ArrayD<f64>) -> ArrayD<f64> {
    let ndim = data.ndim();
    let max = data.map_axis(ndarray::Axis(ndim - 1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))).insert_axis(ndarray::Axis(ndim - 1));
    let shifted = data - &max;
    let exp = shifted.mapv(|v| v.exp());
    let sum = exp.sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
    exp / sum
}

impl Tensor {
    pub fn softmax(&self) -> Self {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Softmax, vec![self.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad();
        let res_data = softmax_forward(&self.data());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SoftmaxBackward {
                input: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

pub fn log_softmax_forward(data: &ArrayD<f64>) -> ArrayD<f64> {
    let ndim = data.ndim();
    let max = data.map_axis(ndarray::Axis(ndim - 1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))).insert_axis(ndarray::Axis(ndim - 1));
    let shifted = data - &max;
    let exp = shifted.mapv(|v| v.exp());
    let sum = exp.sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
    shifted - sum.mapv(|v| v.ln())
}

// --- LogSoftmax ---
pub struct LogSoftmaxBackward {
    pub input: Tensor,
}

impl BackwardFn for LogSoftmaxBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.input.data();
        let ndim = x.ndim();
        let max = x.map_axis(ndarray::Axis(ndim - 1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))).insert_axis(ndarray::Axis(ndim - 1));
        let exp = (&x - &max).mapv(|v| v.exp());
        let sum = exp.sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
        let softmax = exp / sum;
        let sum_grad = grad_output.sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
        vec![grad_output - &softmax * sum_grad]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn op_name(&self) -> &'static str {
        "LogSoftmax"
    }
}

impl Tensor {
    pub fn log_softmax(&self) -> Self {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::LogSoftmax, vec![self.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad();
        let data = self.data();
        let ndim = data.ndim();
        let max = data.map_axis(ndarray::Axis(ndim - 1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b))).insert_axis(ndarray::Axis(ndim - 1));
        let shifted = &data - &max;
        let exp = shifted.mapv(|v| v.exp());
        let sum = exp.sum_axis(ndarray::Axis(ndim - 1)).insert_axis(ndarray::Axis(ndim - 1));
        let log_sum_exp = sum.mapv(|v| v.ln());
        let res_data = shifted - log_sum_exp;

        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(LogSoftmaxBackward {
                input: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

#[derive(Clone)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    LogSoftmax,
}

impl Activation {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Relu => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::Gelu => x.gelu(),
            Activation::LogSoftmax => x.log_softmax(),
        }
    }
}
