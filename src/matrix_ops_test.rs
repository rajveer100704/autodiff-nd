#[cfg(test)]
mod phase4_matrix_ops {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_matmul_shape() {
        // (2, 3) @ (3, 4) => (2, 4)
        let a = Tensor::new(vec![1.0; 6], &[2, 3]);
        let b = Tensor::new(vec![1.0; 12], &[3, 4]);
        let c = a.matmul(&b);
        assert_eq!(c.data().shape(), &[2, 4]);
    }

    #[test]
    fn test_matmul_values() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.matmul(&b);
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (got, exp) in c.data().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*got, *exp, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_matmul_backward_input_grad() {
        // dL/dA = dL/dC @ B^T
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
        a.set_requires_grad(true);
        let c = a.clone().matmul(&b);
        c.sum().backward();
        // grad of A: every row of a gets B's row sum
        let ag = a.grad();
        assert_abs_diff_eq!(ag[[0, 0]], 5.0, epsilon = 1e-7); // col sum of B row 0
        assert_abs_diff_eq!(ag[[0, 1]], 9.0, epsilon = 1e-7); // col sum of B row 1
    }

    #[test]
    fn test_broadcast_add_bias() {
        // (3, 4) + bias(4,) => (3, 4) with broadcast
        let x = Tensor::new(vec![0.0; 12], &[3, 4]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        b.set_requires_grad(true);
        let y = x + b.clone();
        y.sum().backward();
        // Each bias element gets summed grad * batch_size
        assert_abs_diff_eq!(b.grad()[[0]], 3.0, epsilon = 1e-7);
        assert_abs_diff_eq!(b.grad()[[3]], 3.0, epsilon = 1e-7);
    }
}
