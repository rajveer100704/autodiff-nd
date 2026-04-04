#[cfg(test)]
mod phase3_activations {
    use autodiff_nd::engine::Tensor;

    use super::*;
    use approx::assert_abs_diff_eq;

    fn numerical_grad(t: &Tensor, f: impl Fn(Tensor) -> Tensor, eps: f64) -> f64 {
        let val = t.data()[[0]];
        let t_plus = Tensor::new(vec![val + eps], &[1]);
        let t_minus = Tensor::new(vec![val - eps], &[1]);
        (f(t_plus).data()[[0]] - f(t_minus).data()[[0]]) / (2.0 * eps)
    }

    #[test]
    fn test_relu_forward() {
        let a = Tensor::new(vec![-1.0, 0.0, 2.0], &[3]);
        let out = a.relu();
        assert_eq!(out.data().as_slice().unwrap(), &[0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::new(vec![2.0], &[1]);
        a.set_requires_grad(true);
        a.clone().relu().backward();
        assert_abs_diff_eq!(a.grad()[[0]], 1.0, epsilon = 1e-7); // positive side
        let b = Tensor::new(vec![-1.0], &[1]);
        b.set_requires_grad(true);
        b.clone().relu().backward();
        assert_abs_diff_eq!(b.grad()[[0]], 0.0, epsilon = 1e-7); // negative side
    }

    #[test]
    fn test_sigmoid_numerical_grad() {
        let a = Tensor::new(vec![0.5], &[1]);
        a.set_requires_grad(true);
        a.clone().sigmoid().backward();
        let num_grad = numerical_grad(&a, |t| t.sigmoid(), 1e-5);
        assert_abs_diff_eq!(a.grad()[[0]], num_grad, epsilon = 1e-5);
    }

    #[test]
    fn test_tanh_numerical_grad() {
        let a = Tensor::new(vec![0.3], &[1]);
        a.set_requires_grad(true);
        a.clone().tanh().backward();
        let num_grad = numerical_grad(&a, |t| t.tanh(), 1e-5);
        assert_abs_diff_eq!(a.grad()[[0]], num_grad, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_numerical_grad() {
        let a = Tensor::new(vec![0.7], &[1]);
        a.set_requires_grad(true);
        a.clone().gelu().backward();
        let num_grad = numerical_grad(&a, |t| t.gelu(), 1e-5);
        assert_abs_diff_eq!(a.grad()[[0]], num_grad, epsilon = 1e-5);
    }
}
