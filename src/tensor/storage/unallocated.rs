use super::r#trait::TensorStorage;

pub struct UnallocatedTensorStorage;

impl TensorStorage for UnallocatedTensorStorage {
    fn get_data(&self) -> Vec<f32> {
        panic!("Cannot read from unallocated tensor")
    }
    
    fn update_data(&self, _data: Vec<f32>) {
        panic!("Cannot update unallocated tensor")
    }
    
    fn size_in_bytes(&self) -> u64 {
        0
    }
    
    fn is_allocated(&self) -> bool {
        false
    }
    
    fn gpu_idx(&self) -> Option<usize> {
        None
    }
    
    fn location_string(&self) -> String {
        "Unallocated Tensor".to_string()
    }
}