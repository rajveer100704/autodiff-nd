use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use ndarray::ArrayD;
use crate::engine::Tensor;

/// Base trait for all neural network modules.
pub trait Module: Send + Sync {
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Returns index-based parameters for internal optimization.
    fn parameters(&self) -> Vec<Tensor>;

    /// Returns named parameters for state_dict serialization.
    /// Default implementation returns empty; modules should override.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Returns a weight-only dictionary of the module state.
    fn state_dict(&self) -> HashMap<String, ArrayD<f64>> {
        self.named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.data()))
            .collect()
    }

    /// Loads weights from a state_dict, with shape validation.
    fn load_state_dict(&self, dict: &HashMap<String, ArrayD<f64>>) {
        for (name, tensor) in self.named_parameters() {
            if let Some(data) = dict.get(&name) {
                let current_shape = tensor.inner().data.read().unwrap().shape().to_vec();
                assert_eq!(
                    current_shape, 
                    data.shape(), 
                    "Shape mismatch for parameter '{}': expected {:?}, got {:?}", 
                    name, current_shape, data.shape()
                );
                *tensor.inner().data.write().unwrap() = data.clone();
            }
        }
    }

    /// Saves the module state to a binary file using Bincode.
    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let dict = self.state_dict();
        bincode::serialize_into(writer, &dict)?;
        Ok(())
    }

    /// Loads the module state from a binary file using Bincode.
    fn load(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dict: HashMap<String, ArrayD<f64>> = bincode::deserialize_from(reader)?;
        self.load_state_dict(&dict);
        Ok(())
    }

    fn train(&self) { self.set_training(true); }
    fn eval(&self) { self.set_training(false); }
    fn set_training(&self, _is_training: bool) {}
}
