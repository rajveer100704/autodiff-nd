#[cfg(test)]
mod tests {
    
    use autodiff_nd::engine::Tensor;
    use autodiff_nd::ops::conv::Conv2d;
    use autodiff_nd::module::Module;

    #[test]
    fn test_conv2d_forward_shape() {
        // (N, C, H, W) = (1, 1, 5, 5)
        let input = Tensor::new(vec![1.0; 25], &[1, 1, 5, 5]);
        // (C_out, C_in, kH, kW) = (1, 1, 3, 3)
        let conv = Conv2d::new(1, 1, (3, 3), (1, 1), (0, 0));
        
        let out = conv.forward(&input);
        // H_out = (5 + 0 - 3)/1 + 1 = 3
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_gradient() {
        // Small test for numerical gradient verification
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ];
        let input = Tensor::new(input_data, &[1, 1, 3, 3]);
        input.set_requires_grad(true);

        let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0));
        // Set weights to 1.0 for simplicity
        {
            let inner = conv.weight.inner();
            let mut w = inner.data.write().unwrap();
            w.fill(1.0);
        }

        let out = conv.forward(&input);
        let loss = out.sum();
        loss.backward();

        let grad_in = input.grad();
        // Manual verification for 3x3 input, 2x2 kernel of 1s, sum of output:
        // Output is 2x2:
        // [1+2+4+5, 2+3+5+6]
        // [4+5+7+8, 5+6+8+9]
        // Total sum = 12 + 16 + 24 + 28 = 80
        // Grad w.r.t input:
        // [1, 2, 1]
        // [2, 4, 2]
        // [1, 2, 1]
        
        assert_eq!(grad_in[[0, 0, 0, 0]], 1.0);
        assert_eq!(grad_in[[0, 0, 1, 1]], 4.0);
        assert_eq!(grad_in[[0, 0, 0, 1]], 2.0);
    }
}
