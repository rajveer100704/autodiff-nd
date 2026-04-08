use criterion::{black_box, criterion_group, criterion_main, Criterion};
use autodiff_nd::{
    engine::{Tensor, binary_cross_entropy},
    linear::Linear,
    module::Module,
    optimizers::{Adam, Optimizer},
};

/// Benchmark a full forward + backward + step cycle for a small MLP.
/// This measures the real training loop cost: allocation, graph build,
/// backward pass, and parameter update combined.
fn bench_mlp_training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");

    // Small net: 64 → 128 → 64 → 1
    group.bench_function("mlp_64_128_64_1", |b| {
        let l1 = Linear::new(64, 128);
        let l2 = Linear::new(128, 64);
        let l3 = Linear::new(64, 1);
        let params: Vec<Tensor> = l1
            .parameters()
            .into_iter()
            .chain(l2.parameters())
            .chain(l3.parameters())
            .collect();
        let mut opt = Adam::new(params, 1e-3, 0.9, 0.999, 1e-8);

        b.iter(|| {
            opt.zero_grad();
            let x = Tensor::new(vec![1.0_f64; 64], &[1, 64]);
            let tgt = Tensor::new(vec![1.0_f64], &[1, 1]);
            let h1 = l1.forward(&x).relu();
            let h2 = l2.forward(&h1).relu();
            let out = l3.forward(&h2).sigmoid();
            let loss = binary_cross_entropy(&out, &tgt);
            black_box(loss.backward());
            opt.step();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_mlp_training_step);
criterion_main!(benches);
