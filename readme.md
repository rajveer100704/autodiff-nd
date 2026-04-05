# Autodiff-nd 🦀📐

<p align="center">
  <img src="assets/logo.png" alt="Autodiff-nd Logo" width="400">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-2021_edition-orange.svg" alt="Rust 2021">
  <img src="https://img.shields.io/badge/status-experimental-yellow.svg" alt="Experimental">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
</p>

**Autodiff-nd** is a lightweight Automatic Differentiation (Autograd) library for Rust,
leveraging the power of `ndarray` for high-performance N-Dimensional tensor operations.

Built from scratch to deeply understand how backpropagation works under the hood —
the same foundational mechanism behind PyTorch, JAX, and every modern deep learning framework.

> ⚠️ **Experimental & Educational** — This crate is intended for learning and experimentation.
> It is **not** optimized for production workloads. APIs may change between versions.

---

## ✨ Features

### 🔢 Core Engine

- **Dynamic computation graph** built automatically during the forward pass
- **Reverse-mode automatic differentiation** via topological sort backpropagation
- **`Tensor` as a reference-counted handle** — `Rc<RefCell<TensorInner>>` — cloning is cheap and graph-safe
- **Gradient accumulation** — reused tensors accumulate gradients from all paths correctly
- **`requires_grad` flag** — fine-grained control over which tensors participate in the graph
- **Broadcast-aware gradient reduction** — gradients are correctly reduced back to the original shape after broadcasting via `reduce_grad()`

### ➕ Arithmetic Ops (all differentiable)

| Op             | API        | Notes                                 |
| -------------- | ---------- | ------------------------------------- |
| Addition       | `a + b`    | Broadcast-safe                        |
| Subtraction    | `a - b`    | Broadcast-safe                        |
| Multiplication | `a * b`    | Element-wise, broadcast-safe          |
| Division       | `a / b`    | Numerically stable (`eps = 1e-12`)    |
| Negation       | `-a`       |                                       |
| Power          | `a.pow(n)` | Scalar exponent                       |
| Square Root    | `a.sqrt()` | Guarded against `1/(2√x)` div-by-zero |
| Exponential    | `a.exp()`  |                                       |
| Natural Log    | `a.ln()`   | Input clamped to `max(x, 1e-12)`      |

### 🧠 Activation Functions (all differentiable)

| Activation | API           | Notes                                                                                           |
| ---------- | ------------- | ----------------------------------------------------------------------------------------------- |
| ReLU       | `x.relu()`    | Subgradient = 0 at exactly 0                                                                    |
| Sigmoid    | `x.sigmoid()` | `σ(x) = 1 / (1 + e⁻ˣ)`                                                                          |
| Tanh       | `x.tanh()`    | `grad = 1 - tanh²(x)`                                                                           |
| GELU       | `x.gelu()`    | Tanh approximation; used in Transformers                                                        |
| Softmax    | `x.softmax()` | Numerically stable (max subtraction); full Jacobian-vector product in backward; **no Rc cycle** |

### 📐 Tensor Operations (all differentiable)

| Op              | API                 | Notes                                                                |
| --------------- | ------------------- | -------------------------------------------------------------------- |
| Global Sum      | `x.sum()`           | Reduces to scalar                                                    |
| Global Mean     | `x.mean()`          | Reduces to scalar                                                    |
| Axis Sum        | `x.sum_axis(axis)`  | Keeps dimension (`keepdims = true`)                                  |
| Matrix Multiply | `x.matmul(&y)`      | 2D `(M,K)×(K,N)`, batched `(B,M,K)×(K,N)`, batched `(B,M,K)×(B,K,N)` |
| Transpose       | `x.transpose()`     | 2D: `(M,N)→(N,M)`; 3D: swaps last two axes `(B,M,N)→(B,N,M)`         |
| Reshape         | `x.reshape(&shape)` | Full grad graph preserved                                            |
| Slice Row       | `x.slice_row(i)`    | Differentiable row indexing; gradient flows back to parent           |

### 📉 Loss Functions

| Loss                      | API                                     |
| ------------------------- | --------------------------------------- |
| Mean Squared Error        | `mse_loss(&pred, &target)`              |
| Categorical Cross-Entropy | `cross_entropy_loss(&logits, &targets)` |
| Binary Cross-Entropy      | `binary_cross_entropy(&pred, &target)`  |

### 🏗️ Neural Network Modules (`Module` trait)

- **`Linear`** — fully connected layer with Kaiming/He weight initialization, zero bias
- **`BatchNorm1d`** — training mode (batch statistics) + inference mode (running statistics); learnable `γ` and `β`; exponential moving average for `running_mean` / `running_var`
- **`Dropout`** — inverted dropout with configurable drop probability `p`; correctly masked and scaled

All modules implement the `Module` trait:

```rust
pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&self);   // sets training mode
    fn eval(&self);    // sets inference mode
}
```

### ⚙️ Optimizers

| Optimizer | API                                | Notes                              |
| --------- | ---------------------------------- | ---------------------------------- |
| SGD       | `SGD::new(params, lr)`             | `w = w - lr * grad`                |
| Adam      | `Adam::new(params, lr, β1, β2, ε)` | Adaptive moments + bias correction |

Both implement the `Optimizer` trait with `step()` and `zero_grad()`.

### 🛠️ Training Utilities

- **`clip_grad_norm(params, max_norm)`** — clips global gradient norm in-place
- **`l2_regularization(params, λ)`** — returns a differentiable L2 penalty tensor (skips biases, penalizes weights only)

---

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
autodiff-nd = "0.1"
```

### XOR — Training a 2-Layer MLP

```rust
use autodiff_nd::{
    engine::Tensor,
    linear::Linear,
    optimizers::{Adam, Optimizer},
    losses::binary_cross_entropy,
    module::Module,
};

fn main() {
    let xs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0],
        vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let ys = vec![0.0, 1.0, 1.0, 0.0];

    let l1 = Linear::new(2, 4);
    let l2 = Linear::new(4, 1);

    let params: Vec<Tensor> = l1.parameters()
        .into_iter().chain(l2.parameters()).collect();
    let mut opt = Adam::new(params, 0.01, 0.9, 0.999, 1e-8);

    for epoch in 0..2000 {
        opt.zero_grad();
        let mut total_loss = Tensor::new(vec![0.0], &);

        for (x, &y) in xs.iter().zip(ys.iter()) {
            let input  = Tensor::new(x.clone(), &);
            let target = Tensor::new(vec![y], &);

            let h   = l1.forward(&input).sigmoid();
            let out = l2.forward(&h).sigmoid();

            total_loss = total_loss + binary_cross_entropy(&out, &target);
        }

        total_loss.backward();
        opt.step();

        if epoch % 200 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, total_loss.data()[]);
        }
    }
}
```

### Manual Gradient Computation

```rust
use autodiff_nd::engine::Tensor;

let x = Tensor::new(vec![3.0], &);
let w = Tensor::new(vec![2.0], &);
let b = Tensor::new(vec![1.0], &);

x.set_requires_grad(true);
w.set_requires_grad(true);
b.set_requires_grad(true);

// y = w * x + b  →  dy/dw = x = 3, dy/dx = w = 2, dy/db = 1
let y = (w.clone() * x.clone()) + b.clone();
y.backward();

println!("dy/dw = {}", w.grad()[]); // 3.0
println!("dy/dx = {}", x.grad()[]); // 2.0
println!("dy/db = {}", b.grad()[]); // 1.0
```

---

## 🧩 What You Can Build

| Model                            | Status                                                   |
| -------------------------------- | -------------------------------------------------------- |
| MLP (any depth)                  | ✅ Ready                                                 |
| Logistic / Multiclass Classifier | ✅ Ready                                                 |
| Autoencoder                      | ✅ Ready                                                 |
| Transformer FFN Block            | ✅ Ready                                                 |
| Self-Attention (single head)     | ✅ Ready — `matmul` + `transpose` + `softmax`            |
| Multi-Head Attention             | ✅ Ready — `reshape` + `transpose` handle head splitting |
| Full Transformer Encoder         | ✅ Ready                                                 |
| RNN / LSTM                       | ⚠️ Needs per-timestep column slicing (`slice_col`)       |
| Conv1d / Conv2d                  | ❌ Needs `im2col` + strided matmul                       |

---

## ⚠️ Known Limitations

These are intentional trade-offs for simplicity and educational clarity:

| Limitation                    | Detail                                                                   |
| ----------------------------- | ------------------------------------------------------------------------ |
| **Single-threaded only**      | Uses `Rc` (not `Arc`), so tensors cannot cross thread boundaries         |
| **No SIMD / GPU**             | All computation runs on CPU via `ndarray`; no BLAS acceleration          |
| **`f64` only**                | No `f32`, `bf16`, or mixed precision support                             |
| **No lazy evaluation**        | Graph is eagerly evaluated; no kernel fusion or XLA-style compilation    |
| **No second-order gradients** | `backward()` is not differentiable; no Hessians or `grad-of-grad`        |
| **`backward()` is one-shot**  | `grad_fn.take()` consumes the graph; cannot call `.backward()` twice     |
| **2D / 3D matmul only**       | No arbitrary batched N-D matrix multiply                                 |
| **No `DataLoader`**           | No built-in dataset batching, shuffling, or parallel loading             |
| **No model serialization**    | Cannot save/load weights to disk yet                                     |
| **No in-place ops**           | All ops return new tensors; no `x += y` style mutation through the graph |

---

## 🔬 Design Decisions

**Why `Rc<RefCell<>>` instead of `Arc<Mutex<>>`?**
Single-threaded access keeps the borrow rules simple and panics loudly at runtime if you
violate them — much easier to debug during learning than silent data races.

**Why recompute `softmax` in `SoftmaxBackward` instead of caching the output?**
Caching the output tensor inside `grad_fn` creates an `Rc` reference cycle
(`output → grad_fn → output`) that Rust's `Rc` cannot break, causing a memory leak
on every forward pass. Recomputing is cheap and eliminates the cycle entirely.

**Why `grad_fn.take()` during backward?**
`.take()` replaces the `Option<Rc<dyn BackwardFn>>` with `None` after reading it.
This eagerly drops `Rc` references as the backward walk proceeds, breaking any
remaining reference cycles and freeing memory as you go.

---
