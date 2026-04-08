use criterion::{black_box, criterion_group, criterion_main, Criterion};
use autodiff_nd::engine::Tensor;

fn bench_matmul_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // 512x512 square matmul
    group.bench_function("2d_512x512", |b| {
        let a = Tensor::new(vec![1.0_f64; 512 * 512], &[512, 512]);
        let bm = Tensor::new(vec![1.0_f64; 512 * 512], &[512, 512]);
        b.iter(|| black_box(a.matmul(&bm)));
    });

    // Batched 3D: batch=32, 128x128
    group.bench_function("3d_batch32_128x128", |b| {
        let a = Tensor::new(vec![1.0_f64; 32 * 128 * 128], &[32, 128, 128]);
        let bm = Tensor::new(vec![1.0_f64; 128 * 128], &[128, 128]);
        b.iter(|| black_box(a.matmul(&bm)));
    });

    group.finish();
}

fn bench_matmul_backward(c: &mut Criterion) {
    c.bench_function("matmul_backward_256x256", |b| {
        b.iter(|| {
            let a = Tensor::new(vec![1.0_f64; 256 * 256], &[256, 256]);
            let bm = Tensor::new(vec![1.0_f64; 256 * 256], &[256, 256]);
            a.set_requires_grad(true);
            bm.set_requires_grad(true);
            let out = a.matmul(&bm);
            let loss = out.sum();
            black_box(loss.backward());
        });
    });
}

criterion_group!(benches, bench_matmul_2d, bench_matmul_backward);
criterion_main!(benches);
