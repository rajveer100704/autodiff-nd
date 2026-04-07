use rand::rng;
use rand_distr::{Distribution, Normal};

use crate::{engine::Tensor, module::Module};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rng();
        // Kaiming/He Initialization: Normal(0, sqrt(2/fan_in))

        let std = (2.0 / in_features as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight_data: Vec<f64> = (0..in_features * out_features)
            .map(|_| normal.sample(&mut rng))
            .collect();

        let weight = Tensor::new(weight_data, &[in_features, out_features]);
        weight.set_requires_grad(true);

        // biases are set to zero
        let bias_data = vec![0.0; out_features];
        let bias = Tensor::new(bias_data, &[out_features]);
        bias.set_requires_grad(true);
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight) + self.bias.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
