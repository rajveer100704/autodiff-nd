#[cfg(test)]
mod phase6_linear_optimizer {
    use autodiff_nd::engine::{Tensor, mse_loss};
    use autodiff_nd::linear::Linear;
    use autodiff_nd::module::Module;
    use autodiff_nd::optimizers::{Adam, Optimizer, SGD};

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_layer_output_shape() {
        let layer = Linear::new(4, 8); // in=4, out=8
        let x = Tensor::new(vec![1.0; 8], &[2, 4]); // batch=2
        let y = layer.forward(&x);
        assert_eq!(y.data().shape(), &[2, 8]);
    }

    #[test]
    fn test_sgd_reduces_loss() {
        // y = W*x + b;  fit a single scalar target
        let layer = Linear::new(1, 1);
        let x = Tensor::new(vec![1.0], &[1, 1]);
        let target = Tensor::new(vec![5.0], &[1, 1]);
        let mut opt = SGD::new(layer.parameters(), 0.1);

        let mut last_loss = f64::MAX;
        for _ in 0..50 {
            opt.zero_grad();
            let pred = layer.forward(&x);
            let loss = mse_loss(&pred, &target);
            loss.backward();
            opt.step();
            last_loss = loss.data()[[0]];
        }
        assert!(last_loss < 0.1, "SGD did not converge: loss={}", last_loss);
    }
    #[test]
    fn test_adam_reduces_loss() {
        // Simple regression: y = 2x + 1
        let layer = Linear::new(1, 1);

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1]);
        let target = Tensor::new(vec![3.0, 5.0, 7.0, 9.0], &[4, 1]);

        let mut opt = Adam::new(layer.parameters(), 0.01, 0.9, 0.999, 1e-8);

        // Initial loss
        let initial_loss = mse_loss(&layer.forward(&x), &target).data()[[0]];

        let mut final_loss = initial_loss;

        for _ in 0..1000 {
            opt.zero_grad();

            let pred = layer.forward(&x);
            let loss = mse_loss(&pred, &target);

            loss.backward();
            opt.step();

            final_loss = loss.data()[[0]];
        }

        assert!(
            final_loss < initial_loss,
            "Adam did not reduce loss: initial={}, final={}",
            initial_loss,
            final_loss
        );

        // Optional stronger check
        assert!(
            final_loss < 0.1,
            "Adam did not converge sufficiently: final={}",
            final_loss
        );
    }

    #[test]
    fn test_zero_grad_clears_grad() {
        let layer = Linear::new(2, 2);
        let x = Tensor::new(vec![1.0, 1.0], &[1, 2]);
        let mut opt = SGD::new(layer.parameters(), 0.1);

        // Two backward passes without zero_grad should not double-count
        opt.zero_grad();
        mse_loss(&layer.forward(&x), &Tensor::new(vec![1.0, 1.0], &[1, 2])).backward();
        let g_before = layer.weight.grad().clone();
        opt.zero_grad();
        mse_loss(&layer.forward(&x), &Tensor::new(vec![1.0, 1.0], &[1, 2])).backward();
        let g_after = layer.weight.grad().clone();
        // Grads should be equal (not doubled)
        assert_abs_diff_eq!(g_before[[0, 0]], g_after[[0, 0]], epsilon = 1e-7);
    }
}
