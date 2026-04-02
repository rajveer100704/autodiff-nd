#[cfg(test)]
mod phase6_linear_optimizer {
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
    fn test_adam_reduces_loss_faster_than_sgd() {
        // Confirm Adam converges in fewer steps on a simple regression
        let make_problem = || {
            let layer = Linear::new(2, 1);
            let x = Tensor::new(vec![1.0, 2.0], &[1, 2]);
            let target = Tensor::new(vec![3.0], &[1, 1]);
            (layer, x, target)
        };

        let steps_needed = |use_adam: bool| -> usize {
            let (layer, x, target) = make_problem();
            let mut opt: Box<dyn Optimizer> = if use_adam {
                Box::new(Adam::new(layer.parameters(), 0.01, 0.9, 0.999, 1e-8))
            } else {
                Box::new(SGD::new(layer.parameters(), 0.01))
            };
            for i in 0..500 {
                opt.zero_grad();
                let loss = mse_loss(&layer.forward(&x), &target);
                loss.backward();
                opt.step();
                if loss.data()[[0]] < 0.01 {
                    return i;
                }
            }
            500
        };

        assert!(steps_needed(true) <= steps_needed(false));
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
