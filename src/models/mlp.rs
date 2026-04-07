use crate::{
    engine::{Activation, Tensor},
    linear::Linear,
    module::Module,
};

pub struct MLP {
    layers: Vec<Box<dyn Module>>,
    hidden_activation: Activation,
    output_activation: Option<Activation>,
}

impl MLP {
    pub fn new(
        layer_sizes: &[usize],
        hidden_activation: Activation,
        output_activation: Option<Activation>,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output layer"
        );

        let mut layers: Vec<Box<dyn Module>> = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let in_f = layer_sizes[i];
            let out_f = layer_sizes[i + 1];

            layers.push(Box::new(Linear::new(in_f, out_f)));
        }

        MLP {
            layers,
            hidden_activation,
            output_activation,
        }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let is_last = i == self.layers.len() - 1;

            // 1. Apply the linear transformation FIRST
            out = layer.forward(&out);

            // 2. Then apply activation
            if !is_last {
                out = self.hidden_activation.apply(&out);
            } else if let Some(act) = &self.output_activation {
                out = act.apply(&out);
            }
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    // Propagate train/eval down to all submodules
    fn set_training(&self, flag: bool) {
        for layer in &self.layers {
            layer.set_training(flag);
        }
    }
}
