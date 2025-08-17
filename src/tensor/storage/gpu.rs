use crate::{gpu::gpu_memory::GPUMemory, utils::expect_msg::ExpectMsg};
use super::r#trait::TensorStorage;

pub struct GpuTensorStorage {
    gpu_idx: usize,
    memory: GPUMemory,
}

impl GpuTensorStorage {
    pub fn new(gpu_idx: usize, memory: GPUMemory) -> Self {
        Self { gpu_idx, memory }
    }
    
    // Direct GPU memory access if needed
    pub fn memory(&self) -> &GPUMemory {
        &self.memory
    }
}

impl TensorStorage for GpuTensorStorage {
    fn get_data(&self) -> Vec<f32> {
        self.memory.read_memory().expect_msg("Failed to read GPU memory")
    }
    
    fn update_data(&self, data: Vec<f32>) {
        let expected_elements = (self.memory.size as usize) / std::mem::size_of::<f32>();
        if data.len() != expected_elements {
            panic!("Input data size mismatch: expected {} elements, got {}", 
                   expected_elements, data.len());
        }
        self.memory.copy_into(&data).expect_msg("Failed to copy data to GPU memory");
    }
    
    fn size_in_bytes(&self) -> u64 {
        self.memory.size
    }
    
    fn is_allocated(&self) -> bool {
        true
    }
    
    fn gpu_idx(&self) -> Option<usize> {
        Some(self.gpu_idx)
    }
    
    fn location_string(&self) -> String {
        format!("GPU {} Tensor", self.gpu_idx)
    }
}