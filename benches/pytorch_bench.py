import torch
import time

def bench_linear(in_f=2048, out_f=2048):
    print(f"\n[PyTorch Linear: {in_f} -> {out_f}]")
    model = torch.nn.Linear(in_f, out_f)
    x = torch.randn(1, in_f, requires_grad=True)

    # Forward
    start = time.time()
    for _ in range(10):
        _ = model(x)
    fw_time = (time.time() - start) / 10
    print(f"   Forward:          {fw_time:.4f}s")

    # Forward + Backward
    start = time.time()
    for _ in range(10):
        model.zero_grad()
        y = model(x)
        y.sum().backward()
    bw_time = (time.time() - start) / 10
    print(f"   Forward+Backward: {bw_time:.4f}s")

def bench_no_grad(in_f=1024, out_f=1024):
    print("\n[PyTorch no_grad() speedup]")
    model = torch.nn.Linear(in_f, out_f)
    x = torch.randn(100, in_f)

    # Eager
    start = time.time()
    for _ in range(100):
        _ = model(x)
    eager_time = (time.time() - start) / 100

    # no_grad
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    nograd_time = (time.time() - start) / 100

    print(f"   Eager Mode:       {eager_time:.4f}s")
    print(f"   no_grad Mode:     {nograd_time:.4f}s")
    print(f"   🚀 Speedup:       {(eager_time / (nograd_time + 1e-9) - 1) * 100:.1f}%")

if __name__ == "__main__":
    bench_linear()
    bench_no_grad()
