use autodiff_nd::{Linear, Module};
use std::fs;

#[test]
fn test_state_dict_roundtrip() {
    let model = Linear::new(10, 5);
    let original_weights = model.weight.data();
    
    let path = "test_model_checkpoint.bin";
    model.save(path).expect("Failed to save model");
    
    let new_model = Linear::new(10, 5);
    new_model.load(path).expect("Failed to load model");
    
    let loaded_weights = new_model.weight.data();
    
    // Assert weights are identical
    for (a, b) in original_weights.iter().zip(loaded_weights.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    
    fs::remove_file(path).ok();
}

#[test]
#[should_panic]
fn test_state_dict_shape_mismatch() {
    let model = Linear::new(10, 5);
    let path = "test_model_mismatch.bin";
    model.save(path).expect("Failed to save model");
    
    let wrong_model = Linear::new(10, 6); // Different output features
    wrong_model.load(path).unwrap(); // Should panic on shape mismatch
    
    fs::remove_file(path).ok();
}
