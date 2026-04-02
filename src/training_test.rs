#[cfg(test)]
mod phase7_training {
    use super::*;

    /// XOR is the canonical non-linearly-separable toy problem.
    /// A 2→4→1 MLP with sigmoid should solve it.
    #[test]
    fn test_mlp_learns_xor() {
        let xs: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

        let l1 = Linear::new(2, 4);
        let l2 = Linear::new(4, 1);

        let params: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).collect();
        let mut opt = Adam::new(params, 0.01, 0.9, 0.999, 1e-8);

        for _epoch in 0..2000 {
            opt.zero_grad();
            let mut total_loss = Tensor::new(vec![0.0], &[1]);
            for (x, &y) in xs.iter().zip(ys.iter()) {
                let inp = Tensor::new(x.to_vec(), &[1, 2]);
                let tgt = Tensor::new(vec![y], &[1, 1]);
                let h = l1.forward(&inp).sigmoid();
                let out = l2.forward(&h).sigmoid();
                total_loss = total_loss + binary_cross_entropy(&out, &tgt);
            }
            total_loss.backward();
            opt.step();
        }

        // Check predictions are correct
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let inp = Tensor::new(x.to_vec(), &[1, 2]);
            let h = l1.forward(&inp).sigmoid();
            let out = l2.forward(&h).sigmoid().data()[[0, 0]];
            let pred = if out > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(pred, y, "XOR failed on input {:?}: got {}", x, out);
        }
    }

    /// Gradient clipping prevents exploding gradients in deep nets.
    #[test]
    fn test_gradient_clipping() {
        let layer = Linear::new(10, 10);
        let x = Tensor::new(vec![1.0; 10], &[1, 10]);
        let t = Tensor::new(vec![10.0; 10], &[1, 10]); // large residual → large grads
        mse_loss(&layer.forward(&x), &t).backward();
        clip_grad_norm(layer.parameters(), 1.0);
        let total_norm: f64 = layer
            .parameters()
            .iter()
            .flat_map(|p| p.grad().iter().map(|g| g * g).collect::<Vec<_>>())
            .sum::<f64>()
            .sqrt();
        assert!(total_norm <= 1.0 + 1e-6, "Clip failed: norm={}", total_norm);
    }

    /// L2 regularization (weight decay) should increase loss on over-fit weights.
    #[test]
    fn test_weight_decay_penalizes_large_weights() {
        let layer = Linear::new(2, 1);
        // Manually set large weights
        layer.weight.0.borrow_mut().data.fill(100.0);
        let x = Tensor::new(vec![1.0, 1.0], &[1, 2]);
        let tgt = Tensor::new(vec![0.0], &[1, 1]);
        let loss_no_reg = mse_loss(&layer.forward(&x), &tgt).data()[[0]];
        let loss_with_reg = (mse_loss(&layer.forward(&x), &tgt)
            + l2_regularization(layer.parameters(), 1e-2))
        .data()[[0]];
        assert!(loss_with_reg > loss_no_reg);
    }

    /// BatchNorm should normalize outputs to ~0 mean and ~1 std during training.
    #[test]
    fn test_batchnorm_normalizes_output() {
        let bn = BatchNorm1d::new(4);
        let x = Tensor::new(
            vec![10.0, 20.0, 30.0, 40.0, 11.0, 22.0, 31.0, 42.0],
            &[2, 4],
        );
        let out = bn.forward(&x, true); // training=true
        let mean: f64 = out.data().mean().unwrap();
        let std: f64 = out.data().std(1.0);
        assert!(mean.abs() < 0.1, "BatchNorm mean not near 0: {}", mean);
        assert!((std - 1.0).abs() < 0.2, "BatchNorm std not near 1: {}", std);
    }

    /// Dropout should zero ~p fraction of outputs during training.
    #[test]
    fn test_dropout_zero_fraction() {
        let dropout = Dropout::new(0.5);
        let x = Tensor::new(vec![1.0; 1000], &[1000]);
        let out = dropout.forward(&x, true);
        let zero_count = out.data().iter().filter(|&&v| v == 0.0).count();
        // Should be roughly 500 ± 60 zeros
        assert!(
            zero_count > 400 && zero_count < 600,
            "Dropout fraction wrong: {} zeros",
            zero_count
        );
    }
}
