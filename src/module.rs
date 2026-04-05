use crate::engine::Tensor;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;

    // Returns all tensors in this module that require gradients
    fn parameters(&self) -> Vec<Tensor>;

    fn set_training(&self, _is_training: bool) {}
}
