# autodiff-nd 🚀

**Implements a full-stack ML system: autograd engine → graph compiler → kernel optimization.**

Autodiff-nd is a ground-up re-engineering of modern deep learning systems, inspired by frameworks like PyTorch and JAX. It implements the full ML systems stack—from reverse-mode autodiff and parallel execution to a compiler-backed graph IR and kernel-level optimizations such as FlashAttention.

The goal is not just functionality, but to understand and reproduce the core design principles behind production ML infrastructure.

---

## ⚡ What Makes This Different

Unlike typical ML projects, autodiff-nd is built as a systems project:

- **Concurrency-first design** using `Arc<RwLock>` (thread-safe autograd)
- **Lazy Execution Engine** with Graph IR and optimization passes
- **Compiler-driven operator fusion** (e.g., Matmul → Add → ReLU)
- **FlashAttention kernel integration** ($O(N^2) \rightarrow O(N)$ memory)
- **Rayon-based parallel execution** for massive CPU throughput

---

## 🏗️ Architecture Deep Dive

### 1. Autograd Engine
- **Reverse-mode autodiff** with dynamic computation graph.
- **Topological traversal** for efficient backward passes.
- **Broadcast-aware gradient reduction** for complex tensor shapes.

### 2. Concurrency Model
- **Thread-safe state**: Every tensor utilizes `Arc<RwLock>` to enable safe multi-threaded execution.
- **Performance Optimized**: Separate locks for data and gradients reduce contention during parallel training.

### 3. Compiler Layer
- **Graph IR**: Lightweight representation of the computation graph.
- **Lazy Tracing**: Scoped tracing blocks enable optimization before execution.
- **Pattern-based fusion**: Automatically identifies and fuses patterns like `Matmul -> Add -> ReLU`.

### 4. Kernel Optimization
- **FlashAttention**: Implemented as a compiler pass.
- **Streaming Softmax**: Avoids materializing the $N \times N$ attention matrix.
- **Complexity**: Reduces memory from $O(N^2) \rightarrow O(N)$, enabling processing of massive sequence lengths.

---

## 🧠 Systems Coverage

This project spans the entire ML systems stack:

| Layer | Implementation |
|:--- |:--- |
| **Framework** | Autograd, Modules, State Serialization, Optimizers |
| **Execution** | Parallel Eager Engine (Rayon), Thread-local Tracing Guards |
| **Compiler** | Graph IR (DAG Representation), Fusion Passes, Topological JIT |
| **Kernel** | FlashAttention (Tiling + Streaming Softmax) |

---

## 📈 Benchmarks

### ⚡ Performance scaling
Measured on a multi-core CPU, showing the impact of kernel-level fusion in FlashAttention. Benchmarks averaged over 10 runs (± stddev) after warmup to ensure cache-stable measurements.

![Time Benchmark](benchmark_time.png)

### 💾 Memory Scaling
FlashAttention reduces memory complexity from $O(N^2)$ to $O(N)$.

![Memory Benchmark](benchmark_memory.png)

> [!NOTE]
> Memory usage is estimated based on tensor dimensions and 64-bit float size.
> All results are CPU-based; GPU acceleration and kernel codegen are left as future work.

| Sequence Length (N) | Eager Attention | FlashAttention | Speedup |
| :--- | :--- | :--- | :--- |
| **512** | ~2.0 MB | **~0.25 MB** | 1.8x |
| **1024** | ~8.0 MB | **~0.50 MB** | 2.5x |
| **2048** | ~32.0 MB | **~1.00 MB** | 3.1x |

---

## 🔁 Reproducibility

### Run the Benchmarks
Generate statistical results (mean/stddev) for performance and memory scaling:
```bash
cargo run --release --example flash_attention
```

### Generate Plots
Use the Python plotting engine to create visual assets:
```bash
python scripts/plot_benchmarks.py
```

---

## 🚀 Future Roadmap

- [ ] **SIMD Optimization**: AVX-512 backend for core kernels.
- [ ] **Distributed Autograd**: Sharding tensors across multiple nodes.
- [ ] **Quantization Pass**: INT8/FP8 compiler passes for low-precision inference.

---

## 🛠️ Performance & Safety
- **Rayon-powered**: Automatic multi-threading.
- **Zero-Panic**: Hardened state management for concurrent access.
- **Efficiency**: Minimal allocations during lazy graph construction.
