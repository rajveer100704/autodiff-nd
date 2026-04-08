use autodiff_nd::{Conv2d, Linear, Tensor, no_grad, Module};
use std::time::Instant;

fn main() {
    println!("📊 autodiff-nd Performance Benchmark 📊");
    println!("========================================\n");

    // 1. Matmul (Linear) Forward vs Forward+Backward
    bench_linear(2048, 2048);

    // 2. Conv2d (im2col + GEMM)
    bench_conv2d(16, 64, 128, (3, 3));

    // 3. no_grad() Speedup
    bench_no_grad(1024, 1024);

    println!("\n✅ Benchmarks complete.");
    println!("Compare these results with PyTorch using 'benches/pytorch_bench.py'");
}

fn bench_linear(in_f: usize, out_f: usize) {
    println!("[Linear Layer: {} -> {}]", in_f, out_f);
    let model = Linear::new(in_f, out_f);
    let x = Tensor::new(vec![0.5; in_f], &[1, in_f]);
    x.set_requires_grad(true);

    // Forward
    let start = Instant::now();
    for _ in 0..10 {
        let _ = model.forward(&x);
    }
    let fw_time = start.elapsed().as_secs_f64() / 10.0;
    println!("   Forward:          {:.4}s", fw_time);

    // Forward + Backward
    let start = Instant::now();
    for _ in 0..10 {
        let y = model.forward(&x);
        y.backward();
        model.parameters().iter().for_each(|p| p.zero_grad());
    }
    let bw_time = start.elapsed().as_secs_f64() / 10.0;
    println!("   Forward+Backward: {:.4}s", bw_time);
    println!("   Backward overhead: {:.1}x", bw_time / fw_time);
}

fn bench_conv2d(batch: usize, in_c: usize, out_c: usize, kernel: (usize, usize)) {
    println!("\n[Conv2d: B={}, C_in={}, C_out={}, K={:?}]", batch, in_c, out_c, kernel);
    let model = Conv2d::new(in_c, out_c, kernel, (1, 1), (1, 1));
    let x = Tensor::new(vec![0.5; batch * in_c * 32 * 32], &[batch, in_c, 32, 32]);
    x.set_requires_grad(true);

    let start = Instant::now();
    for _ in 0..5 {
        let _ = model.forward(&x);
    }
    println!("   Forward (Rayon):  {:.4}s", start.elapsed().as_secs_f64() / 5.0);
}

fn bench_no_grad(in_f: usize, out_f: usize) {
    println!("\n[Inference Mode: no_grad() speedup]");
    let model = Linear::new(in_f, out_f);
    let x = Tensor::new(vec![0.1; in_f], &[100, in_f]);
    x.set_requires_grad(true);

    // Eager (with graph)
    let start = Instant::now();
    for _ in 0..100 {
        let _ = model.forward(&x);
    }
    let eager_time = start.elapsed().as_secs_f64() / 100.0;

    // no_grad (optimized)
    let start = Instant::now();
    for _ in 0..100 {
        no_grad(|| {
            let _ = model.forward(&x);
        });
    }
    let nograd_time = start.elapsed().as_secs_f64() / 100.0;

    println!("   Eager Mode:       {:.4}s", eager_time);
    println!("   no_grad Mode:     {:.4}s", nograd_time);
    println!("   🚀 Speedup:       {:.1}%", (eager_time / nograd_time - 1.0) * 100.0);
}
