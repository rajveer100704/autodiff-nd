use ndarray::ArrayD;
use crate::engine::Tensor;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.clone() - target.clone();
    diff.pow(2.0).mean()
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let x_data = logits.data();
    let batch = targets.len();
    let num_classes = x_data.shape()[1];
    let mut losses = Vec::with_capacity(batch);
    for (i, &class) in targets.iter().enumerate() {
        let mut mask = vec![0.0; num_classes];
        mask[class] = 1.0;
        let mask_t = Tensor::new(mask, &[num_classes]);
        let row = logits.slice_row(i);
        let log_row = row.softmax().ln();
        let correct_log_prob = (log_row * mask_t).sum();
        losses.push(-correct_log_prob);
    }
    let mut total = losses[0].clone();
    for l in losses.iter().skip(1) {
        total = total + l.clone();
    }
    let n = Tensor::new(vec![batch as f64], &[1]);
    total / n
}

pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let one = Tensor::from_array(ArrayD::ones(
        pred.inner().data.read().unwrap().raw_dim(),
    ));
    let term1 = target.clone() * pred.ln();
    let term2 = (one.clone() - target.clone()) * (one - pred.clone()).ln();
    -(term1 + term2).mean()
}
