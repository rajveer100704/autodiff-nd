use ndarray::ArrayD;
use std::collections::HashMap;
use crate::compiler::{Graph, NodeId, OpKind};
use crate::ops::{imaging, activation, matrix, attention};

pub struct Executor;

impl Executor {
    /// Executes the graph given a set of input values for Input nodes.
    pub fn execute(graph: &Graph, input_values: HashMap<NodeId, ArrayD<f64>>) -> HashMap<NodeId, ArrayD<f64>> {
        let sorted = graph.topo_sort();
        let mut values = input_values;

        for &node_id in &sorted {
            if values.contains_key(&node_id) {
                continue; // Already provided (Input node)
            }

            let node = &graph.nodes[node_id];
            let inputs: Vec<ArrayD<f64>> = node.inputs.iter()
                .map(|id| values.get(id).expect("Input value missing during execution").clone())
                .collect();

            let result = match node.op {
                OpKind::Input => unreachable!("Inputs should be in the initial values"),
                OpKind::Add => &inputs[0] + &inputs[1],
                OpKind::Sub => &inputs[0] - &inputs[1],
                OpKind::Mul => &inputs[0] * &inputs[1],
                OpKind::Div => &inputs[0] / &inputs[1],
                OpKind::Matmul => matrix::matmul_forward(&inputs[0], &inputs[1]),
                OpKind::ReLU => inputs[0].mapv(|x| x.max(0.0)),
                OpKind::Conv2d => {
                    let attrs = node.attrs.as_ref().expect("Conv2d missing attributes in IR");
                    let bias = if inputs.len() > 2 { Some(&inputs[2]) } else { None };
                    imaging::conv2d_forward(
                        &inputs[0], 
                        &inputs[1], 
                        bias, 
                        attrs.stride, 
                        attrs.padding
                    )
                },
                OpKind::Reshape | OpKind::Flatten => {
                    inputs[0].clone().into_shape_with_order(node.shape.clone()).expect("Reshape fail during execution")
                },
                OpKind::LogSoftmax => activation::log_softmax_forward(&inputs[0]),
                OpKind::Softmax => activation::softmax_forward(&inputs[0]),
                OpKind::Transpose => matrix::transpose_forward(&inputs[0]),
                OpKind::Scale => {
                    let scale = node.attrs.as_ref().and_then(|a| a.scale).unwrap_or(1.0);
                    &inputs[0] * scale
                },
                OpKind::FlashAttention => {
                    let attrs = node.attrs.as_ref().expect("FlashAttention missing attributes");
                    let scale = attrs.scale.unwrap_or(1.0);
                    
                    // FlashAttention requires 2D arrays (N, d)
                    let q = inputs[0].clone().into_dimensionality::<ndarray::Ix2>().expect("Q must be 2D");
                    let k = inputs[1].clone().into_dimensionality::<ndarray::Ix2>().expect("K must be 2D");
                    let v = inputs[2].clone().into_dimensionality::<ndarray::Ix2>().expect("V must be 2D");
                    
                    let res = attention::flash_attention_forward(&q, &k, &v, scale, 64);
                    res.into_dyn()
                },
                OpKind::FusedMatmulAddRelu => {
                    let xw = matrix::matmul_forward(&inputs[0], &inputs[1]);
                    let xwb = &xw + &inputs[2];
                    xwb.mapv(|x| x.max(0.0))
                },
            };

            values.insert(node_id, result);
        }

        values
    }
}
