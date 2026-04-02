#[cfg(test)]
mod phase1_autograd_basics {
    use crate::engine::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_add_forward() {
        let a = Tensor::new(vec![2.0], &[1]);
        let b = Tensor::new(vec![3.0], &[1]);
        let c = a + b;
        assert_eq!(c.data()[[0]], 5.0);
    }

    #[test]
    fn test_mul_forward() {
        let a = Tensor::new(vec![3.0], &[1]);
        let b = Tensor::new(vec![4.0], &[1]);
        let c = a * b;
        assert_eq!(c.data()[[0]], 12.0);
    }

    #[test]
    fn test_add_backward_grad() {
        // z = a + b  =>  dz/da = 1, dz/db = 1
        let a = Tensor::new(vec![2.0], &[1]);
        let b = Tensor::new(vec![3.0], &[1]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let z = a.clone() + b.clone();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(b.grad()[[0]], 1.0, epsilon = 1e-7);
    }

    #[test]
    fn test_mul_backward_grad() {
        // z = a * b  =>  dz/da = b = 4, dz/db = a = 3
        let a = Tensor::new(vec![3.0], &[1]);
        let b = Tensor::new(vec![4.0], &[1]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let z = a.clone() * b.clone();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 4.0, epsilon = 1e-7);
        assert_abs_diff_eq!(b.grad()[[0]], 3.0, epsilon = 1e-7);
    }

    #[test]
    fn test_chained_ops_grad() {
        // z = (a + b) * c  =>  dz/da = c, dz/dc = a+b
        let a = Tensor::new(vec![2.0], &[1]);
        let b = Tensor::new(vec![3.0], &[1]);
        let c = Tensor::new(vec![5.0], &[1]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        c.set_requires_grad(true);
        let z = (a.clone() + b.clone()) * c.clone();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 5.0, epsilon = 1e-7);
        assert_abs_diff_eq!(b.grad()[[0]], 5.0, epsilon = 1e-7);
        assert_abs_diff_eq!(c.grad()[[0]], 5.0, epsilon = 1e-7); // a+b = 5
    }

    #[test]
    fn test_no_grad_doesnt_propagate() {
        let a = Tensor::new(vec![2.0], &[1]); // requires_grad = false
        let b = Tensor::new(vec![3.0], &[1]);
        b.set_requires_grad(true);
        let z = a + b.clone();
        z.backward();
        // b should still get grad because the output required it
        assert_abs_diff_eq!(b.grad()[[0]], 1.0, epsilon = 1e-7);
    }

    #[test]
    fn test_grad_accumulates_on_reuse() {
        // z = a * a  =>  dz/da = 2*a = 4 (grad accumulates from two paths)
        let a = Tensor::new(vec![2.0], &[1]);
        a.set_requires_grad(true);
        let z = a.clone() * a.clone();
        z.backward();
        assert_abs_diff_eq!(a.grad()[[0]], 4.0, epsilon = 1e-7);
    }
}
