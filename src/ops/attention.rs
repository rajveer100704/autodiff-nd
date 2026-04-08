use ndarray::{Array2, s};
use std::f64;

/// Implements a memory-efficient FlashAttention forward pass for the CPU.
/// This kernel uses tiling and online softmax to achieve O(N) memory complexity 
/// by avoiding the materialization of the N x N attention matrix.
pub fn flash_attention_forward(
    q: &Array2<f64>, // (N, d)
    k: &Array2<f64>, // (N, d)
    v: &Array2<f64>, // (N, d)
    scale: f64,
    block_size: usize,
) -> Array2<f64> {
    let (n, d) = q.dim();
    let mut output = Array2::<f64>::zeros((n, d));

    // For each query row i (or block of rows)
    for i in 0..n {
        let qi = q.row(i);

        // Running statistics for online softmax
        let mut m_i = f64::NEG_INFINITY; // Running max
        let mut l_i = 0.0;               // Running sum of exponentials
        let mut o_i = vec![0.0; d];      // Running weighted sum accumulator

        // For each block of keys/values
        for start in (0..n).step_by(block_size) {
            let end = (start + block_size).min(n);

            let k_block = k.slice(s![start..end, ..]);
            let v_block = v.slice(s![start..end, ..]);

            // 1. Compute scores: S_ij = (qi * kj.T) * scale
            let mut scores = Vec::with_capacity(end - start);
            for kb in k_block.outer_iter() {
                let dot = qi.dot(&kb) * scale;
                scores.push(dot);
            }

            // 2. Compute block max for numerical stability
            let m_ij = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            // 3. Update global max and running denominators
            let m_new = m_i.max(m_ij);

            // 4. Compute stabilized exponentials for this block
            let mut p = Vec::with_capacity(scores.len());
            for &s_val in &scores {
                p.push((s_val - m_new).exp());
            }

            // Adjustment factor for previous block statistics
            let alpha = (m_i - m_new).exp();
            
            // 5. Update running sum of exps (denominator)
            let p_sum = p.iter().sum::<f64>();
            let l_new = alpha * l_i + p_sum;

            // 6. Update output accumulator: o_i = o_i * alpha + p * v_block
            for j in 0..d {
                o_i[j] *= alpha;
            }
            
            for (idx, vb) in v_block.outer_iter().enumerate() {
                let weight = p[idx];
                for j in 0..d {
                    o_i[j] += weight * vb[j];
                }
            }

            // Commit new statistics
            m_i = m_new;
            l_i = l_new;
        }

        // Final normalization for the row
        for j in 0..d {
            output[[i, j]] = o_i[j] / l_i;
        }
    }

    output
}
