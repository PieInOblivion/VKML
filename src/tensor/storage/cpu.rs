use std::sync::RwLock;
use crate::utils::expect_msg::ExpectMsg;
use super::r#trait::TensorStorage;

pub struct CpuTensorStorage {
    data: RwLock<Vec<f32>>,
}

impl CpuTensorStorage {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }
    
    pub fn with_zeros(size: usize) -> Self {
        Self {
            data: RwLock::new(vec![0.0; size]),
        }
    }
    
    // Optional: Keep these for CPU-specific direct access if needed
    // But now they return RwLock guards instead of RefCell guards
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<Vec<f32>> {
        self.data.read().expect_msg("Failed to acquire read lock on CPU tensor data")
    }
    
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<Vec<f32>> {
        self.data.write().expect_msg("Failed to acquire write lock on CPU tensor data")
    }
}

impl TensorStorage for CpuTensorStorage {
    fn get_data(&self) -> Vec<f32> {
        self.data.read()
            .expect_msg("Failed to acquire read lock on CPU tensor data")
            .clone()
    }
    
    fn update_data(&self, data: Vec<f32>) {
        let mut guard = self.data.write()
            .expect_msg("Failed to acquire write lock on CPU tensor data");
        
        let expected_elements = guard.len();
        if data.len() != expected_elements {
            panic!("Input data size mismatch: expected {} elements, got {}", 
                   expected_elements, data.len());
        }
        *guard = data;
    }
    
    fn size_in_bytes(&self) -> u64 {
        let guard = self.data.read()
            .expect_msg("Failed to acquire read lock on CPU tensor data");
        (guard.len() * std::mem::size_of::<f32>()) as u64
    }
    
    fn is_allocated(&self) -> bool {
        true
    }
    
    fn gpu_idx(&self) -> Option<usize> {
        None
    }
    
    fn location_string(&self) -> String {
        "CPU Tensor".to_string()
    }
}