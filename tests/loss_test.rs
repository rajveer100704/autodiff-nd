#[cfg(test)]
mod phase5_losses {
    use autodiff_nd::engine::{Tensor, binary_cross_entropy, cross_entropy_loss, mse_loss};

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_loss_zero_on_perfect() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let loss = mse_loss(&pred, &target);
        assert_abs_diff_eq!(loss.data()[[0]], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_mse_loss_backward() {
        let pred = Tensor::new(vec![2.0, 4.0], &[2]);
        let target = Tensor::new(vec![1.0, 2.0], &[2]);
        pred.set_requires_grad(true);
        let loss = mse_loss(&pred, &target);
        loss.backward();
        // dMSE/d(pred_i) = 2*(pred_i - target_i)/N
        assert_abs_diff_eq!(pred.grad()[[0]], 1.0, epsilon = 1e-7); // 2*(2-1)/2
        assert_abs_diff_eq!(pred.grad()[[1]], 2.0, epsilon = 1e-7); // 2*(4-2)/2
    }

    #[test]
    fn test_cross_entropy_loss_range() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let targets = vec![2usize]; // class index 2 is correct
        let loss = cross_entropy_loss(&logits, &targets);
        // loss must be positive
        assert!(loss.data()[[0]] > 0.0);
    }

    #[test]
    fn test_cross_entropy_backward_numerical() {
        // Verify cross-entropy grad via finite differences
        let logits = Tensor::new(vec![1.0, 2.0, 0.5], &[1, 3]);
        logits.set_requires_grad(true);
        let targets = vec![1usize];
        let loss = cross_entropy_loss(&logits, &targets);
        loss.backward();
        // Sum of gradients of softmax cross-entropy should equal 0 (closed form)
        let grad_sum: f64 = logits.grad().iter().sum();
        assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let pred = Tensor::new(vec![0.9], &[1]);
        let target = Tensor::new(vec![1.0], &[1]);
        pred.set_requires_grad(true);
        let loss = binary_cross_entropy(&pred, &target);
        loss.backward();
        // Correct prediction → small loss
        assert!(loss.data()[[0]] < 0.2);
        assert!(pred.grad()[[0]] < 0.0); // gradient pushes pred higher
    }
}
