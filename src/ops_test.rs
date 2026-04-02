#[cfg(test)]
mod phase2_ops {
    use std::ops::Neg;

    use crate::engine::Tensor;

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sub_backward() {
        // z = a - b  =>  dz/da = 1, dz/db = -1
        let a = Tensor::new(vec![5.0], &[1]);
        let b = Tensor::new(vec![3.0], &[1]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let z = a.clone() - b.clone();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(b.grad()[[0]], -1.0, epsilon = 1e-7);
    }

    #[test]
    fn test_neg_backward() {
        // z = -a  =>  dz/da = -1
        let a = Tensor::new(vec![3.0], &[1]);
        a.set_requires_grad(true);
        let z = a.clone().neg();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], -1.0, epsilon = 1e-7);
    }

    #[test]
    fn test_pow_backward() {
        // z = a^3  =>  dz/da = 3*a^2 = 12.0 when a=2
        let a = Tensor::new(vec![2.0], &[1]);
        a.set_requires_grad(true);
        let z = a.clone().pow(3.0);
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 12.0, epsilon = 1e-7);
    }

    #[test]
    fn test_sum_backward() {
        // z = sum([a, b, c])  =>  grad all = 1
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        a.set_requires_grad(true);
        let z = a.clone().sum();
        z.backward();
        assert!(a.grad().iter().all(|&g| (g - 1.0).abs() < 1e-7));
    }

    #[test]
    fn test_exp_backward() {
        // z = exp(a)  =>  dz/da = exp(a)
        let a = Tensor::new(vec![1.0], &[1]);
        a.set_requires_grad(true);
        let z = a.clone().exp();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], std::f64::consts::E, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_backward() {
        // z = mean([a,b,c]) =>  dz/da_i = 1/N
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        a.set_requires_grad(true);
        let z = a.clone().mean();
        z.backward();
        assert!(a.grad().iter().all(|&g| (g - 1.0 / 3.0).abs() < 1e-7));
    }

    #[test]
    fn test_ln_backward() {
        // z = ln(a)  =>  dz/da = 1/a
        let a = Tensor::new(vec![2.0], &[1]);
        a.set_requires_grad(true);
        let z = a.clone().ln();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 0.5, epsilon = 1e-7);
    }
}
