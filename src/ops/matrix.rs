use std::sync::Arc;

use ndarray::{Array2, ArrayD};
use crate::engine::{Tensor, BackwardFn};
use crate::compiler::{context, OpKind};

// --- Matmul ---
pub struct MatmulBackward {
    pub parents: Vec<Tensor>,
}

impl BackwardFn for MatmulBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let a = self.parents[0].inner().data.read().unwrap().clone();
        let b = self.parents[1].inner().data.read().unwrap().clone();
        let ndim = grad_output.ndim();

        if ndim == 2 {
            let g = grad_output.into_dimensionality::<ndarray::Ix2>().unwrap();
            let a2 = a.into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2 = b.into_dimensionality::<ndarray::Ix2>().unwrap();
            let grad_a = g.dot(&b2.t()).into_dyn();
            let grad_b = a2.t().dot(&g).into_dyn();
            vec![grad_a, grad_b]
        } else if ndim == 3 {
            let g3 = grad_output.into_dimensionality::<ndarray::Ix3>().unwrap();
            let batch = g3.shape()[0];
            let b_is_2d = b.ndim() == 2;

            use rayon::prelude::*;

            if b_is_2d {
                let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let results: Vec<(Array2<f64>, Array2<f64>)> = (0..batch)
                    .into_par_iter()
                    .map(|i| {
                        let gi = g3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                        let ai = a.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                        (gi.dot(&b2.t()), ai.t().dot(&gi))
                    })
                    .collect();

                let mut grad_a_out = ndarray::Array3::<f64>::zeros((batch, a.shape()[1], a.shape()[2]));
                let mut grad_b_out = ArrayD::zeros(b.raw_dim());
                {
                    let mut gb2 = grad_b_out.view_mut().into_dimensionality::<ndarray::Ix2>().unwrap();
                    for (i, (da, db)) in results.into_iter().enumerate() {
                        grad_a_out.index_axis_mut(ndarray::Axis(0), i).assign(&da);
                        gb2 += &db;
                    }
                }
                vec![grad_a_out.into_dyn(), grad_b_out]
            } else {
                let b3 = b.view().into_dimensionality::<ndarray::Ix3>().unwrap();
                let results: Vec<(Array2<f64>, Array2<f64>)> = (0..batch)
                    .into_par_iter()
                    .map(|i| {
                        let gi = g3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                        let ai = a.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                        let bi = b3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                        (gi.dot(&bi.t()), ai.t().dot(&gi))
                    })
                    .collect();

                let mut grad_a_out = ndarray::Array3::<f64>::zeros((batch, a.shape()[1], a.shape()[2]));
                let mut grad_b_out = ndarray::Array3::<f64>::zeros((batch, b.shape()[1], b.shape()[2]));

                for (i, (da, db)) in results.into_iter().enumerate() {
                    grad_a_out.index_axis_mut(ndarray::Axis(0), i).assign(&da);
                    grad_b_out.index_axis_mut(ndarray::Axis(0), i).assign(&db);
                }
                vec![grad_a_out.into_dyn(), grad_b_out.into_dyn()]
            }
        } else {
            panic!("matmul only supports 2D and 3D tensors, got {}D", ndim);
        }
    }
    fn parents(&self) -> Vec<Tensor> {
        self.parents.clone()
    }
    fn op_name(&self) -> &'static str {
        "Matmul"
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        if context::is_tracing() {
            // Shape calculation for lazy tracing
            let a_shape = &self.shape;
            let b_shape = &rhs.shape;
            let res_shape = if a_shape.len() == 2 && b_shape.len() == 2 {
                vec![a_shape[0], b_shape[1]]
            } else if a_shape.len() == 3 && b_shape.len() == 2 {
                vec![a_shape[0], a_shape[1], b_shape[1]]
            } else if a_shape.len() == 3 && b_shape.len() == 3 {
                vec![a_shape[0], a_shape[1], b_shape[2]]
            } else {
                panic!("Incompatible shapes for matmul: {:?} and {:?}", a_shape, b_shape);
            };

            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Matmul, vec![self.node_id(), rhs.node_id()], res_shape.clone())
            });
            return Tensor::from_lazy(id, res_shape);
        }

        let req_grad = self.requires_grad() || rhs.requires_grad();
        let a = self.data();
        let b = rhs.data();
        let res_data = matmul_forward(&a, &b);
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(MatmulBackward {
                parents: vec![self.clone(), rhs.clone()],
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

pub fn matmul_forward(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    use rayon::prelude::*;
    match (a.ndim(), b.ndim()) {
        (2, 2) => {
            let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            a2.dot(&b2).into_dyn()
        }
        (3, 2) => {
            let a3 = a.view().into_dimensionality::<ndarray::Ix3>().unwrap();
            let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let batch = a3.shape()[0];
            let slices: Vec<Array2<f64>> = (0..batch).into_par_iter().map(|i| {
                let ai = a3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                ai.dot(&b2)
            }).collect();
            let mut out = ndarray::Array3::<f64>::zeros((batch, a3.shape()[1], b2.shape()[1]));
            for (i, s) in slices.into_iter().enumerate() { out.index_axis_mut(ndarray::Axis(0), i).assign(&s); }
            out.into_dyn()
        }
        (3, 3) => {
            let a3 = a.view().into_dimensionality::<ndarray::Ix3>().unwrap();
            let b3 = b.view().into_dimensionality::<ndarray::Ix3>().unwrap();
            let batch = a3.shape()[0];
            let slices: Vec<Array2<f64>> = (0..batch).into_par_iter().map(|i| {
                let ai = a3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                let bi = b3.index_axis(ndarray::Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
                ai.dot(&bi)
            }).collect();
            let mut out = ndarray::Array3::<f64>::zeros((batch, a3.shape()[1], b3.shape()[2]));
            for (i, s) in slices.into_iter().enumerate() { out.index_axis_mut(ndarray::Axis(0), i).assign(&s); }
            out.into_dyn()
        }
        (a_ndim, b_ndim) => panic!("matmul: unsupported {}D @ {}D", a_ndim, b_ndim),
    }
}

// --- Transpose ---
pub struct TransposeBackward {
    pub parent: Tensor,
}

impl BackwardFn for TransposeBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let ndim = grad_output.ndim();
        let transposed = if ndim == 2 {
            let g2 = grad_output.into_dimensionality::<ndarray::Ix2>().unwrap();
            g2.t().to_owned().into_dyn()
        } else if ndim == 3 {
            grad_output.permuted_axes(ndarray::IxDyn(&[0, 2, 1])).as_standard_layout().to_owned().into_dyn()
        } else {
            panic!("transpose only supports 2D and 3D tensors");
        };
        vec![transposed]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Transpose"
    }
}

pub fn transpose_forward(data: &ArrayD<f64>) -> ArrayD<f64> {
    let ndim = data.ndim();
    if ndim == 2 {
        let d2 = data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        d2.t().to_owned().into_dyn()
    } else if ndim == 3 {
        data.clone().permuted_axes(ndarray::IxDyn(&[0, 2, 1])).as_standard_layout().to_owned().into_dyn()
    } else {
        panic!("transpose only supports 2D and 3D tensors");
    }
}

impl Tensor {
    pub fn transpose(&self) -> Self {
        if context::is_tracing() {
            let res_shape = if self.shape.len() == 2 {
                vec![self.shape[1], self.shape[0]]
            } else if self.shape.len() == 3 {
                vec![self.shape[0], self.shape[2], self.shape[1]]
            } else {
                panic!("transpose only supports 2D and 3D tensors");
            };
            
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Transpose, vec![self.node_id()], res_shape.clone())
            });
            return Tensor::from_lazy(id, res_shape);
        }

        let req_grad = self.requires_grad();
        let res_data = transpose_forward(&self.data());
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(TransposeBackward {
                parent: self.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- Reshape ---
pub struct ReshapeBackward {
    pub parent: Tensor,
    pub input_shape: Vec<usize>,
}

impl BackwardFn for ReshapeBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output.into_shape_with_order(self.input_shape.clone()).unwrap()]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "Reshape"
    }
}

impl Tensor {
    pub fn reshape(&self, shape: &[usize]) -> Self {
        if context::is_tracing() {
            let id = context::with_graph(|g| {
                crate::compiler::add_op(g, OpKind::Reshape, vec![self.node_id()], shape.to_vec())
            });
            return Tensor::from_lazy(id, shape.to_vec());
        }

        let req_grad = self.requires_grad();
        let data = self.data();
        let res_data = data.into_shape_with_order(shape).expect("Reshape fail").into_dyn();
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(ReshapeBackward {
                parent: self.clone(),
                input_shape: self.shape.clone(),
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }
}

// --- SliceRow ---
pub struct SliceRowBackward {
    pub parent: Tensor,
    pub row_idx: usize,
}

impl BackwardFn for SliceRowBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let parent_shape = self.parent.inner().data.read().unwrap().shape().to_vec();
        let mut grad = ArrayD::zeros(parent_shape);
        grad.index_axis_mut(ndarray::Axis(0), self.row_idx).assign(&grad_output);
        vec![grad]
    }
    fn parents(&self) -> Vec<Tensor> {
        vec![self.parent.clone()]
    }
    fn op_name(&self) -> &'static str {
        "SliceRow"
    }
}

impl Tensor {
    pub fn slice_row(&self, i: usize) -> Self {
        let req_grad = self.requires_grad();
        let inner = self.inner();
        let data = inner.data.read().unwrap();
        let res_data = data.index_axis(ndarray::Axis(0), i).to_owned().into_dyn();
        drop(data);
        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(SliceRowBackward {
                parent: self.clone(),
                row_idx: i,
            }))
        } else {
            None
        };
        Tensor::make_result(res_data, req_grad, grad_fn)
    }

    pub fn flatten(&self) -> Self {
        let inner = self.inner();
        let data = inner.data.read().unwrap();
        let shape = data.shape();
        let batch = shape[0];
        let rest = shape[1..].iter().product();
        drop(data);
        self.reshape(&[batch, rest])
    }
}
