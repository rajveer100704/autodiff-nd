use ndarray::ArrayD;


pub mod context;
pub mod exec;
pub mod pass;

pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Input,      // Eager tensor used as input to lazy graph
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    ReLU,
    Conv2d,
    Reshape,
    Flatten,
    LogSoftmax,
    FusedMatmulAddRelu,
    Softmax,
    Transpose,
    Scale,
    FlashAttention,
}

#[derive(Debug, Clone)]
pub struct NodeAttrs {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub scale: Option<f64>, // For Scale and FlashAttention
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: OpKind,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub attrs: Option<NodeAttrs>,
    pub value: Option<ArrayD<f64>>, // Execution caching
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: OpKind, inputs: Vec<NodeId>, shape: Vec<usize>, attrs: Option<NodeAttrs>) -> NodeId {
        let id = self.nodes.len();
        let node = Node {
            id,
            op,
            inputs,
            shape,
            attrs,
            value: None,
        };
        self.nodes.push(node);
        id
    }
}

pub fn add_op(graph: &mut Graph, op: OpKind, inputs: Vec<NodeId>, shape: Vec<usize>) -> NodeId {
    graph.add_node(op, inputs, shape, None)
}

impl Graph {
    /// Topological sort of nodes required for outputs
    pub fn topo_sort(&self) -> Vec<NodeId> {
        let mut sorted = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        for &out_id in &self.outputs {
            self.topo_visit(out_id, &mut visited, &mut sorted);
        }
        sorted
    }

    fn topo_visit(&self, id: NodeId, visited: &mut Vec<bool>, sorted: &mut Vec<NodeId>) {
        if visited[id] { return; }
        visited[id] = true;
        for &input_id in &self.nodes[id].inputs {
            self.topo_visit(input_id, visited, sorted);
        }
        sorted.push(id);
    }

    /// Debug print of the graph structure
    pub fn print(&self) {
        println!("------- Lazy Graph IR -------");
        for node in &self.nodes {
            println!("  [%{}] {:?} [inputs: {:?}] [shape: {:?}]", 
                node.id, node.op, node.inputs, node.shape);
        }
        println!("  Outputs: {:?}", self.outputs);
        println!("-----------------------------");
    }
}
