#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use autodiff_nd::engine::{Tensor, CustomFunction, Context};

    struct SquareFunc;
    impl CustomFunction for SquareFunc {
        fn forward(&self, inputs: &[ArrayD<f64>], ctx: &mut Context) -> Vec<ArrayD<f64>> {
            let x = &inputs[0];
            ctx.save_for_backward(x.clone());
            vec![x.mapv(|v| v * v)]
        }

        fn backward(&self, ctx: &Context, grad_outputs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
            let x = &ctx.saved_tensors[0];
            let g = &grad_outputs[0];
            // grad = 2x * g
            vec![2.0 * x * g]
        }
    }

    #[test]
    fn test_custom_square() {
        let x = Tensor::new(vec![3.0], &[1]);
        x.set_requires_grad(true);

        let results = Tensor::apply(SquareFunc, &[x.clone()]);
        let y = &results[0];
        assert_eq!(y.data()[[0]], 9.0);

        y.backward();
        // dy/dx = 2*3 = 6
        assert_eq!(x.grad()[[0]], 6.0);
    }
}
