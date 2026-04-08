use crate::compiler::{Graph, NodeId, OpKind};
use std::collections::HashMap;

pub struct PassManager;

impl PassManager {
    /// Fuses Matmul -> Add -> ReLU into a single node.
    pub fn fuse_matmul_add_relu(graph: &Graph) -> Graph {
        let mut new_graph = Graph::new();
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut skipped: Vec<bool> = vec![false; graph.nodes.len()];

        for i in 0..graph.nodes.len() {
            if skipped[i] { continue; }

            let node = &graph.nodes[i];
            
            // Check for ReLU(Add(Matmul(X, W), B))
            if node.op == OpKind::ReLU {
                let add_id = node.inputs[0];
                let add_node = &graph.nodes[add_id];
                
                if add_node.op == OpKind::Add {
                    let mat_id = add_node.inputs[0];
                    let mat_node = &graph.nodes[mat_id];
                    
                    if mat_node.op == OpKind::Matmul {
                        // We found it!
                        // Inputs: Matmul(0), Matmul(1), Add(1) [assuming bias is second input of Add]
                        let inputs = vec![
                            *id_map.get(&mat_node.inputs[0]).unwrap(), // X
                            *id_map.get(&mat_node.inputs[1]).unwrap(), // W
                            *id_map.get(&add_node.inputs[1]).unwrap(), // B
                        ];
                        
                        let fused_id = new_graph.add_node(
                            OpKind::FusedMatmulAddRelu, 
                            inputs, 
                            node.shape.clone(), 
                            None
                        );
                        id_map.insert(node.id, fused_id);
                        skipped[add_id] = true;
                        skipped[mat_id] = true;
                        continue;
                    }
                }
            }

            // Standard node copy if no pattern matched
            let new_inputs: Vec<NodeId> = node.inputs.iter()
                .map(|id| *id_map.get(id).expect("Input ID not mapped in Optimizer"))
                .collect();
            
            let new_id = new_graph.add_node(node.op, new_inputs, node.shape.clone(), node.attrs.clone());
            id_map.insert(node.id, new_id);
        }

        // Map outputs
        new_graph.outputs = graph.outputs.iter()
            .map(|id| *id_map.get(id).expect("Output ID not mapped in Optimizer"))
            .collect();

        new_graph
    }

    /// Fuses attention patterns into FlashAttention.
    pub fn fuse_flash_attention(graph: &Graph) -> Graph {
        let mut new_graph = Graph::new();
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut skipped: Vec<bool> = vec![false; graph.nodes.len()];

        for i in 0..graph.nodes.len() {
            if skipped[i] { continue; }

            let node = &graph.nodes[i];
            
            // Try to match attention pattern: Matmul(Softmax(Scale(Matmul(Q, Transpose(K)))), V)
            if node.op == OpKind::Matmul {
                if let Some(am) = Self::match_attention(i, graph) {
                    // Success! Fuse it.
                    let inputs = vec![
                        *id_map.get(&am.q).unwrap(),
                        *id_map.get(&am.k).unwrap(),
                        *id_map.get(&am.v).unwrap(),
                    ];
                    
                    let fused_id = new_graph.add_node(
                        OpKind::FlashAttention,
                        inputs,
                        node.shape.clone(),
                        Some(crate::compiler::NodeAttrs {
                            stride: (0, 0),
                            padding: (0, 0),
                            scale: Some(am.scale),
                        })
                    );
                    
                    println!("[Compiler] Fused attention pattern -> FlashAttention node %{}", fused_id);
                    
                    id_map.insert(node.id, fused_id);
                    // Mark consumed internal nodes as skipped
                    for &internal_id in &am.internal_nodes {
                        skipped[internal_id] = true;
                    }
                    continue;
                }
            }

            // Normal copy
            let new_inputs: Vec<NodeId> = node.inputs.iter()
                .map(|id| *id_map.get(id).expect("Input ID not mapped"))
                .collect();
            let new_id = new_graph.add_node(node.op, new_inputs, node.shape.clone(), node.attrs.clone());
            id_map.insert(node.id, new_id);
        }

        new_graph.outputs = graph.outputs.iter().map(|id| *id_map.get(id).unwrap()).collect();
        new_graph
    }

    fn match_attention(node_id: usize, graph: &Graph) -> Option<AttentionMatch> {
        let node = &graph.nodes[node_id];
        
        // Root: Matmul(ScoreMatrix, V)
        if node.op != OpKind::Matmul || node.inputs.len() != 2 { return None; }
        let score_id = node.inputs[0];
        let v_id = node.inputs[1];
        
        let score_node = &graph.nodes[score_id];
        if score_node.op != OpKind::Softmax { return None; }
        
        // Softmax(Scale? (Matmul(Q, K_T)))
        let maybe_scale_id = score_node.inputs[0];
        let maybe_scale_node = &graph.nodes[maybe_scale_id];
        
        let (matmul_qk_id, scale) = if maybe_scale_node.op == OpKind::Scale {
            (maybe_scale_node.inputs[0], maybe_scale_node.attrs.as_ref().and_then(|a| a.scale).unwrap_or(1.0))
        } else {
            (maybe_scale_id, 1.0)
        };
        
        let matmul_qk = &graph.nodes[matmul_qk_id];
        if matmul_qk.op != OpKind::Matmul || matmul_qk.inputs.len() != 2 { return None; }
        
        let q_id = matmul_qk.inputs[0];
        let maybe_k_t_id = matmul_qk.inputs[1];
        let maybe_k_t_node = &graph.nodes[maybe_k_t_id];
        
        if maybe_k_t_node.op != OpKind::Transpose { return None; }
        let k_id = maybe_k_t_node.inputs[0];
        
        Some(AttentionMatch {
            q: q_id,
            k: k_id,
            v: v_id,
            scale,
            internal_nodes: vec![score_id, maybe_scale_id, matmul_qk_id, maybe_k_t_id],
        })
    }
}

struct AttentionMatch {
    q: NodeId,
    k: NodeId,
    v: NodeId,
    scale: f64,
    internal_nodes: Vec<NodeId>,
}
