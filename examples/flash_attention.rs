use autodiff_nd::{Tensor, compiler::context, compiler::pass::PassManager, compiler::exec::Executor};
use std::collections::HashMap;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 { return 0.0; }
    let variance = data.iter().map(|value| {
        let diff = mean - (*value);
        diff * diff
    }).sum::<f64>() / (data.len() as f64);
    variance.sqrt()
}

fn benchmark_n(n: usize, d: usize, iterations: usize, warmup: usize) -> (f64, f64, f64, f64) {
    let scale = 1.0 / (d as f64).sqrt();
    let q = Tensor::randn(&[n, d]);
    let k = Tensor::randn(&[n, d]);
    let v = Tensor::randn(&[n, d]);

    // Graph Construction (Eager/Default)
    let ((q_id, k_id, v_id, _), graph_eager) = context::lazy(|| {
        let q_node = q.node_id();
        let k_node = k.node_id();
        let v_node = v.node_id();
        let scores = q.matmul(&k.transpose());
        let scaled = scores.scale(scale);
        let weights = scaled.softmax();
        let res = weights.matmul(&v);
        (q_node, k_node, v_node, res.node_id())
    });

    // Graph Construction (Flash/Optimized)
    let ((fq_id, fk_id, fv_id, _), graph_flash) = context::lazy(|| {
        let q_node = q.node_id();
        let k_node = k.node_id();
        let v_node = v.node_id();
        let scores = q.matmul(&k.transpose());
        let scaled = scores.scale(scale);
        let weights = scaled.softmax();
        let res = weights.matmul(&v);
        (q_node, k_node, v_node, res.node_id())
    });
    let _ = PassManager::fuse_flash_attention(&graph_flash);

    let mut eager_times = Vec::new();
    let mut flash_times = Vec::new();

    let mut inputs = HashMap::new();
    inputs.insert(q_id, q.data());
    inputs.insert(k_id, k.data());
    inputs.insert(v_id, v.data());

    let mut f_inputs = HashMap::new();
    f_inputs.insert(fq_id, q.data());
    f_inputs.insert(fk_id, k.data());
    f_inputs.insert(fv_id, v.data());

    // Warmup
    for _ in 0..warmup {
        let _ = Executor::execute(&graph_eager, inputs.clone());
        let _ = Executor::execute(&graph_flash, f_inputs.clone());
    }

    // Benchmark Eager
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Executor::execute(&graph_eager, inputs.clone());
        eager_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Benchmark Flash
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Executor::execute(&graph_flash, f_inputs.clone());
        flash_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let e_mean = mean(&eager_times);
    let e_std = stddev(&eager_times, e_mean);
    let f_mean = mean(&flash_times);
    let f_std = stddev(&flash_times, f_mean);

    (e_mean, e_std, f_mean, f_std)
}

fn main() {
    println!("=== autodiff-nd: System Benchmarking (Elite Version) ===");
    println!("Statistical setup: 5 warmup, 10 iterations per N\n");

    let mut file = File::create("bench_results.csv").unwrap();
    writeln!(file, "N,eager_mean,eager_std,flash_mean,flash_std,eager_mem,flash_mem").unwrap();

    let d = 64;
    let n_values = [128, 256, 512, 1024, 2048];
    let iterations = 10;
    let warmup = 5;

    for &n in &n_values {
        print!("Benchmarking N={:<5} ... ", n);
        std::io::stdout().flush().unwrap();
        
        let (e_m, e_s, f_m, f_s) = benchmark_n(n, d, iterations, warmup);
        
        // Memory estimate (f64 = 8 bytes)
        // Eager: O(N^2) for attention matrix
        let eager_mem = (n * n * 8) as f64 / (1024.0 * 1024.0);
        // Flash: O(N*d) for streaming
        let flash_mem = (n * d * 8) as f64 / (1024.0 * 1024.0);

        writeln!(file, "{},{},{},{},{},{},{}", n, e_m, e_s, f_m, f_s, eager_mem, flash_mem).unwrap();
        println!("Eager: {:.2}ms (±{:.2}), Flash: {:.2}ms (±{:.2})", e_m, e_s, f_m, f_s);
    }

    println!("\n[Success] Results saved to bench_results.csv");
}
