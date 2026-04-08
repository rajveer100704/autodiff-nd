use autodiff_nd::{
    Activation, Conv2d, Linear, Module, Tensor,
    nll_loss, no_grad,
};
use ndarray::{ArrayD, IxDyn};
use std::time::Instant;

/// A flagship CNN for MNIST-like data.
/// Architecture: Conv2D -> ReLU -> Conv2D -> Flatten -> Linear -> LogSoftmax
struct MnistCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc: Linear,
}

impl MnistCNN {
    fn new() -> Self {
        Self {
            // (in_c, out_c, kernel, stride, padding)
            conv1: Conv2d::new(1, 8, (3, 3), (1, 1), (1, 1)),   // 28x28 -> 28x28
            conv2: Conv2d::new(8, 16, (3, 3), (2, 2), (1, 1)),  // 28x28 -> 14x14
            fc: Linear::new(16 * 14 * 14, 10),                 // 3136 -> 10
        }
    }
}

impl Module for MnistCNN {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv1.forward(x).relu();
        let x = self.conv2.forward(&x).relu();
        let x = x.flatten();
        let x = self.fc.forward(&x);
        x.log_softmax()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.fc.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = Vec::new();
        for (n, t) in self.conv1.named_parameters() { p.push((format!("conv1.{}", n), t)); }
        for (n, t) in self.conv2.named_parameters() { p.push((format!("conv2.{}", n), t)); }
        for (n, t) in self.fc.named_parameters() { p.push((format!("fc.{}", n), t)); }
        p
    }
}

fn main() {
    println!("🚀 Starting autodiff-nd Showcase: MNIST CNN 🚀");
    println!("----------------------------------------------");

    let model = MnistCNN::new();
    let batch_size = 16;
    let epochs = 5;
    let lr = 0.01;

    // Generate synthetic MNIST-like data (28x28)
    // In a real scenario, this would be loaded from idx files.
    let x_train = Tensor::new(vec![0.5; batch_size * 1 * 28 * 28], &[batch_size, 1, 28, 28]);
    
    // One-hot targets (10 classes)
    let mut y_data = vec![0.0; batch_size * 10];
    for i in 0..batch_size { y_data[i * 10 + (i % 10)] = 1.0; } // target = i % 10
    let y_train = Tensor::new(y_data, &[batch_size, 10]);

    let start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        
        // --- Forward Pass ---
        let output = model.forward(&x_train);
        let loss = nll_loss(&output, &y_train);

        // --- Backward Pass ---
        loss.backward();

        // --- Optimizer Step (Manual SGD) ---
        let params = model.parameters();
        for p in params {
            let inner = p.inner();
            let mut data = inner.data.write().unwrap();
            let grad = inner.grad.read().unwrap();
            *data -= &(&*grad * lr);
            drop(grad);
            p.zero_grad();
        }

        // --- Metrics ---
        let loss_val = loss.data().sum();
        let accuracy = no_grad(|| {
            let out_data = output.data();
            let mut correct = 0;
            for i in 0..batch_size {
                let row = out_data.index_axis(ndarray::Axis(0), i);
                let mut max_idx = 0;
                let mut max_val = row[0];
                for (j, &val) in row.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
                if max_idx == (i % 10) { correct += 1; }
            }
            (correct as f64 / batch_size as f64) * 100.0
        });

        println!(
            "Epoch {}: Loss = {:.4} | Accuracy = {:.1}% | Time = {:?}",
            epoch, loss_val, accuracy, epoch_start.elapsed()
        );

        // --- Checkpointing ---
        if epoch % 2 == 0 {
            model.save("mnist_cnn_checkpoint.bin").expect("Failed to save checkpoint");
            println!("   [Checkpoint] Model saved to mnist_cnn_checkpoint.bin");
        }
    }

    println!("----------------------------------------------");
    println!("✅ Training complete! Total time: {:?}", start.elapsed());
    println!("Framework capability verified: Conv2D, Autograd, Reshape, NLLLoss, Serialization.");
}
