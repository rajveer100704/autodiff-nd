use crate::engine::Tensor;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;

    // Returns all tensors in this module that require gradients
    fn parameters(&self) -> Vec<Tensor>;

    // // Helper to zero out gradients for all parameters before a new backward pass
    // fn zero_grad(&self) {
    //     for p in self.parameters() {
    //         let mut inner = p.0.borrow_mut();
    //         inner.grad.fill(0.0);
    //     }
    // }
}
