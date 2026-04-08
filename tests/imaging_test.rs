#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn, array};
    use autodiff_nd::ops::imaging::{im2col_single, col2im_single};

    #[test]
    fn test_im2col_simple() {
        // 1 Channel, 3x3 Input
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let input = array![
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]
        ].into_dyn();

        // 2x2 Kernel, Stride 1, Padding 0
        // Expected Output:
        // [1 2 4 5]  <- Window 1
        // [2 3 5 6]  <- Window 2
        // [4 5 7 8]  <- Window 3
        // [5 6 8 9]  <- Window 4
        // Transposed for GEMM (C*kH*kW, H_out*W_out):
        // [1 2 4 5]
        // [2 3 5 6]
        // [4 5 7 8]
        // [5 6 8 9]

        let col = im2col_single(input.view(), (2, 2), (1, 1), (0, 0));
        
        assert_eq!(col.shape(), &[4, 4]);
        
        // Check window 1 (column 0)
        assert_eq!(col[[0, 0]], 1.0);
        assert_eq!(col[[1, 0]], 2.0);
        assert_eq!(col[[2, 0]], 4.0);
        assert_eq!(col[[3, 0]], 5.0);

        // Check window 4 (column 3)
        assert_eq!(col[[0, 3]], 5.0);
        assert_eq!(col[[1, 3]], 6.0);
        assert_eq!(col[[2, 3]], 8.0);
        assert_eq!(col[[3, 3]], 9.0);
    }

    #[test]
    fn test_col2im_reconstruction() {
        // Test that col2im correctly accumulates
        let channels = 1;
        let h_in = 3;
        let w_in = 3;
        let input = ArrayD::from_elem(IxDyn(&[channels, h_in, w_in]), 1.0);
        
        let kernel_size = (2, 2);
        let stride = (1, 1);
        let padding = (0, 0);

        let col = im2col_single(input.view(), kernel_size, stride, padding);
        let reconstructed = col2im_single(&col, (channels, h_in, w_in), kernel_size, stride, padding);

        // In a 3x3 with 2x2 kernel:
        // Center pixel (1,1) is covered by all 4 windows.
        // Corner pixels are covered by 1 window.
        // Edges are covered by 2 windows.
        
        assert_eq!(reconstructed[[0, 0, 0]], 1.0);
        assert_eq!(reconstructed[[0, 1, 1]], 4.0);
        assert_eq!(reconstructed[[0, 0, 1]], 2.0);
    }
}
