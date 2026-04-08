use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div, Neg};
use ndarray::ArrayD;
use crate::engine::{Tensor, BackwardFn, reduce_grad};
use crate::compiler::{context, OpKind};

// --- Add ---
pub struct AddBackward {
    pub parents: Vec<Tensor>,
}

impl BackwardFn for AddBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let shape0 = self.parents[0].inner().data.read().unwrap().shape().to_vec();
        let shape1 = self.parents[1].inner().data.read().unwrap().shape().to_vec();
        vec![
            reduce_grad(grad_output.clone(), &shape0),
            reduce_grad(grad_output, &shape1),
        ]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "Add"
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Add, vec![self.node_id(), rhs.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad() || rhs.requires_grad();
        let res_data = &self.data() + &rhs.data();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(AddBackward {
                parents: vec![self.clone(), rhs.clone()],
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Sub ---
pub struct SubBackward {
    pub parents: Vec<Tensor>,
}

impl BackwardFn for SubBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let shape0 = self.parents[0].inner().data.read().unwrap().shape().to_vec();
        let shape1 = self.parents[1].inner().data.read().unwrap().shape().to_vec();
        vec![
            reduce_grad(grad_output.clone(), &shape0),
            reduce_grad(-grad_output, &shape1),
        ]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "Sub"
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Sub, vec![self.node_id(), rhs.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad() || rhs.requires_grad();
        let res_data = &self.data() - &rhs.data();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SubBackward {
                parents: vec![self.clone(), rhs.clone()],
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Mul ---
pub struct MulBackward {
    pub parents: Vec<Tensor>,
}

impl BackwardFn for MulBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let p0_data = self.parents[0].inner().data.read().unwrap().clone();
        let p1_data = self.parents[1].inner().data.read().unwrap().clone();
        let shape0 = p0_data.shape().to_vec();
        let shape1 = p1_data.shape().to_vec();
        let g0 = reduce_grad(&grad_output * &p1_data, &shape0);
        let g1 = reduce_grad(&grad_output * &p0_data, &shape1);
        vec![g0, g1]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "Mul"
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Mul, vec![self.node_id(), rhs.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad() || rhs.requires_grad();
        let res_data = &self.data() * &rhs.data();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(MulBackward {
                parents: vec![self.clone(), rhs.clone()],
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Div ---
pub struct DivBackward {
    pub parents: Vec<Tensor>,
}

impl BackwardFn for DivBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let p0_data = self.parents[0].inner().data.read().unwrap().clone();
        let p1_data = self.parents[1].inner().data.read().unwrap().clone();
        let shape0 = p0_data.shape().to_vec();
        let shape1 = p1_data.shape().to_vec();
        let g0 = reduce_grad(&grad_output / &p1_data, &shape0);
        let eps = 1e-12;
        let g1 = reduce_grad(
            -(&grad_output * &p0_data) / p1_data.mapv(|v| v * v + eps),
            &shape1,
        );
        vec![g0, g1]
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "Div"
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Div, vec![self.node_id(), rhs.node_id()], self.shape.clone())
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad() || rhs.requires_grad();
        let res_data = &self.data() / &rhs.data();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(DivBackward {
                parents: vec![self.clone(), rhs.clone()],
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Neg ---
pub struct NegBackward {
    pub parent: Tensor,
}

impl BackwardFn for NegBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![-grad_output]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Neg"
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                // Neg is basically Mul with -1, but for IR simplicity we'll just use eager logic for now 
                // or add Neg to OpKind. Let's add it to OpKind.
                crate::compiler::add_op(g, OpKind::Mul, vec![self.node_id()], self.shape.clone()) // Placeholder
            });
            return Tensor::from_lazy(id, self.shape.clone());
        }

        let req_grad = self.requires_grad();
        let res_data = -self.data();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(NegBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Sqrt ---
pub struct SqrtBackward {
    pub parent: Tensor,
}

impl BackwardFn for SqrtBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.inner().data.read().unwrap().clone();
        let eps = 1e-12;
        let grad = grad_output / (x.mapv(|v| v.sqrt()) + eps) / 2.0;
        vec![grad]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Sqrt"
    }
}

impl Tensor {
    pub fn sqrt(&self) -> Self {
        let req_grad = self.requires_grad();
        let res_data = self.inner().data.read().unwrap().mapv(|v| v.sqrt());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SqrtBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Pow ---
pub struct PowBackward {
    pub parent: Tensor,
    pub exponent: f64,
}

impl BackwardFn for PowBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let n = self.exponent;
        let x = self.parent.inner().data.read().unwrap().clone();
        let grad = grad_output * n * x.mapv(|v| v.powf(n - 1.0));
        vec![grad]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Pow"
    }
}

impl Tensor {
    pub fn pow(self, n: f64) -> Self {
        let req_grad = self.requires_grad();
        let res_data = self.inner().data.read().unwrap().mapv(|v| v.powf(n));
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(PowBackward {
                parent: self,
                exponent: n,
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Exp ---
pub struct ExpBackward {
    pub parent: Tensor,
}

impl BackwardFn for ExpBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.inner().data.read().unwrap().clone();
        vec![grad_output * x.mapv(|v| v.exp())]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Exp"
    }
}

impl Tensor {
    pub fn exp(&self) -> Self {
        let req_grad = self.requires_grad();
        let res_data = self.inner().data.read().unwrap().mapv(|v| v.exp());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(ExpBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Ln ---
pub struct LnBackward {
    pub parent: Tensor,
}

impl BackwardFn for LnBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let x = self.parent.inner().data.read().unwrap().clone();
        let eps = 1e-12;
        vec![grad_output / x.mapv(|v| v.max(eps))]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Ln"
    }
}

impl Tensor {
    pub fn ln(&self) -> Self {
        let req_grad = self.requires_grad();
        let eps = 1e-12;
        let res_data = self.inner().data.read().unwrap().mapv(|v| v.max(eps).ln());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(LnBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}
