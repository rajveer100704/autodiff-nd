use std::sync::Arc;
use ndarray::{ArrayD, IxDyn, Array2, Axis};
use crate::engine::{Tensor, BackwardFn};
use crate::ops::imaging::{batch_im2col, col2im_single};
use crate::module::Module;
use crate::compiler::{context, OpKind};
use crate::ops::imaging;
use ndarray::linalg::Dot;

pub struct Conv2dBackward {
    pub x: Tensor,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl BackwardFn for Conv2dBackward {
    fn backward(&self, grad_output: ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let dout = grad_output.into_dimensionality::<ndarray::Ix4>().unwrap();
        let (batch, out_channels, out_h, out_w) = dout.dim();
        let weight_data = self.weight.data();
        let out_c = weight_data.shape()[0];
        let in_c_kh_kw: usize = weight_data.shape()[1..].iter().product();
        let weight_2d = weight_data.to_owned().into_shape_with_order((out_c, in_c_kh_kw)).unwrap();
        
        let mut grad_input = ArrayD::zeros(self.x.shape());
        let mut grad_weight = Array2::<f64>::zeros((out_c, in_c_kh_kw));
        let mut grad_bias = self.bias.as_ref().map(|_| ArrayD::zeros(IxDyn(&[out_c])));

        let cols = batch_im2col(&self.x.data(), (self.weight.shape()[2], self.weight.shape()[3]), self.stride, self.padding);

        for i in 0..batch {
            let dout_i = dout.index_axis(Axis(0), i).into_dimensionality::<ndarray::Ix3>().unwrap();
            let dout_i_reshaped = dout_i.to_owned().into_shape_with_order((out_channels, out_h * out_w)).unwrap();
            let col_i = &cols[i];

            grad_weight += &dout_i_reshaped.dot(&col_i.t());

            if let Some(ref mut gb) = grad_bias {
                let sum_spatial = dout_i_reshaped.sum_axis(Axis(1));
                *gb += &sum_spatial.into_dyn();
            }

            let dcol_i = weight_2d.t().dot(&dout_i_reshaped);
            let parent_shape = self.x.shape();
            let dx_i = col2im_single(
                &dcol_i,
                (parent_shape[1], parent_shape[2], parent_shape[3]),
                (self.weight.shape()[2], self.weight.shape()[3]),
                self.stride,
                self.padding
            );
            grad_input.index_axis_mut(Axis(0), i).assign(&dx_i);
        }

        let mut grads = vec![
            grad_input, 
            grad_weight.into_shape_with_order(self.weight.shape()).unwrap().into_dyn()
        ];
        if let Some(gb) = grad_bias {
            grads.push(gb);
        }
        grads
    }

    fn parents(&self) -> Vec<Tensor> {
        let mut p = vec![self.x.clone(), self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn op_name(&self) -> &'static str {
        "Conv2D"
    }
}

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub kernel_size: (usize, usize),
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;
        let std = (2.0 / fan_in as f64).sqrt();
        
        use rand::rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rng();
        let normal = Normal::new(0.0, std).unwrap();
        
        let weight_data: Vec<f64> = (0..out_channels * fan_in)
            .map(|_| normal.sample(&mut rng))
            .collect();
            
        let weight = Tensor::new(weight_data, &[out_channels, in_channels, kh, kw]);
        weight.set_requires_grad(true);

        let bias = Some(Tensor::new(vec![0.0; out_channels], &[out_channels]));
        if let Some(ref b) = bias { b.set_requires_grad(true); }

        Self {
            weight,
            bias,
            stride,
            padding,
            kernel_size,
            in_channels,
            out_channels,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if context::is_tracing() {
            let h = x.shape()[2];
            let w = x.shape()[3];
            let oh = (h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
            let ow = (w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
            let res_shape = vec![x.shape()[0], self.out_channels, oh, ow];

            let id = context::with_graph(|g| {
                g.add_node(OpKind::Conv2d, vec![x.node_id(), self.weight.node_id()], res_shape.clone(), Some(crate::compiler::NodeAttrs {
                    stride: self.stride,
                    padding: self.padding,
                    scale: None,
                }))
            });
            return Tensor::from_lazy(id, res_shape);
        }

        let x_data = x.data();
        let w_data = self.weight.data();
        let b_opt = self.bias.as_ref().map(|b| b.data());
        
        let req_grad = x.requires_grad() || self.weight.requires_grad() || self.bias.as_ref().map_or(false, |b| b.requires_grad());
        
        let res_data = imaging::conv2d_forward(
            &x_data, &w_data, b_opt.as_ref(), self.stride, self.padding
        );

        let grad_fn: Option<Arc<dyn BackwardFn + Send + Sync>> = if req_grad {
            Some(Arc::new(Conv2dBackward {
                x: x.clone(),
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                stride: self.stride,
                padding: self.padding,
            }))
        } else {
            None
        };
        
        Tensor::make_result(res_data, req_grad, grad_fn)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}
