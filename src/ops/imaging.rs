use ndarray::{ArrayD, Ix4, Array2, Array3, ArrayViewD, Axis};
use rayon::prelude::*;

/// Transforms a single spatial tensor (C, H, W) into a column-format matrix 
/// (C * kH * kW, H_out * W_out) suitable for GEMM.
/// This implementation uses explicit contiguous loops for maximum cache efficiency.
pub fn im2col_single(
    input: ArrayViewD<f64>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array2<f64> {
    let input3 = input.into_dimensionality::<ndarray::Ix3>().expect("Input must be 3D (C, H, W)");
    let (channels, h_in, w_in) = input3.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut col = Array2::<f64>::zeros((channels * kh * kw, h_out * w_out));

    // Nested loops for contiguous output construction
    for c in 0..channels {
        for ky in 0..kh {
            for kx in 0..kw {
                let row_idx = c * kh * kw + ky * kw + kx;
                for y in 0..h_out {
                    for x in 0..w_out {
                        let iy = (y * sh) as isize - ph as isize + ky as isize;
                        let ix = (x * sw) as isize - pw as isize + kx as isize;

                        if iy >= 0 && iy < h_in as isize && ix >= 0 && ix < w_in as isize {
                            col[[row_idx, y * w_out + x]] = input3[[c, iy as usize, ix as usize]];
                        }
                        // Else: padded zero (already zero-initialized)
                    }
                }
            }
        }
    }
    col
}

/// Accumulates gradients from column space (C * kH * kW, H_out * W_out) back to spatial space (C, H, W).
/// Handles overlapping windows via additive accumulation.
pub fn col2im_single(
    col: &Array2<f64>,
    output_shape: (usize, usize, usize), // (C, H, W)
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array3<f64> {
    let (channels, h_in, w_in) = output_shape;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut img = Array3::<f64>::zeros((channels, h_in, w_in));

    for c in 0..channels {
        for ky in 0..kh {
            for kx in 0..kw {
                let row_idx = c * kh * kw + ky * kw + kx;
                for y in 0..h_out {
                    for x in 0..w_out {
                        let iy = (y * sh) as isize - ph as isize + ky as isize;
                        let ix = (x * sw) as isize - pw as isize + kx as isize;

                        if iy >= 0 && iy < h_in as isize && ix >= 0 && ix < w_in as isize {
                            img[[c, iy as usize, ix as usize]] += col[[row_idx, y * w_out + x]];
                        }
                    }
                }
            }
        }
    }
    img
}

/// Batch version of im2col, parallelized across the batch dimension with Rayon.
pub fn batch_im2col(
    input: &ArrayD<f64>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Vec<Array2<f64>> {
    let input4 = input.view().into_dimensionality::<ndarray::Ix4>().expect("Input must be 4D (N, C, H, W)");
    let batch_size = input4.shape()[0];

    (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let slice = input4.index_axis(ndarray::Axis(0), i);
            im2col_single(slice.into_dyn(), kernel_size, stride, padding)
        })
        .collect()
}

/// Centralized Conv2d forward pass logic (im2col + GEMM).
pub fn conv2d_forward(
    input: &ArrayD<f64>,
    weight: &ArrayD<f64>,
    bias: Option<&ArrayD<f64>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> ArrayD<f64> {
    let input4 = input.view().into_dimensionality::<Ix4>().expect("Conv2D input must be 4D");
    let weight4 = weight.view().into_dimensionality::<Ix4>().expect("Conv2D weight must be 4D");
    let (batch, _in_c, h_in, w_in) = input4.dim();
    let (out_c, _in_c_idx, kh, kw) = weight4.dim();
    
    let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;

    let cols = batch_im2col(input, (kh, kw), stride, padding);
    let weight_flat = weight4.to_owned().into_shape_with_order((out_c, _in_c_idx * kh * kw)).unwrap();

    let mut output = ndarray::Array4::<f64>::zeros((batch, out_c, h_out, w_out));

    for i in 0..batch {
        let res_i = weight_flat.dot(&cols[i]);
        let res_i_reshaped = res_i.into_shape_with_order((out_c, h_out, w_out)).unwrap();
        output.index_axis_mut(Axis(0), i).assign(&res_i_reshaped);
    }

    let mut res = output.into_dyn();
    if let Some(b) = bias {
        let b_view = b.view().into_shape_with_order((1, out_c, 1, 1)).unwrap();
        res += &b_view;
    }
    res
}
