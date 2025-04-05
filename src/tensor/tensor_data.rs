use std::cell::{Ref, RefCell, RefMut};

use crate::{dataloader::error::VKMLEngineError, gpu::gpu_memory::GPUMemory};

pub enum TensorData {
    CPU(RefCell<Vec<f32>>),
    GPU { gpu_idx: usize, memory: GPUMemory },
    Unallocated,
}

impl TensorData {
    pub fn get_data(&self) -> Result<Vec<f32>, VKMLEngineError> {
        match self {
            TensorData::CPU(data) => Ok(data.borrow().clone()),
            TensorData::GPU { gpu_idx: _, memory } => memory
                .read_memory()
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string())),
            TensorData::Unallocated => Err(VKMLEngineError::VulkanLoadError(
                "Cannot read from unallocated tensor".to_string(),
            )),
        }
    }

    pub fn borrow_cpu_data(&self) -> Result<Ref<Vec<f32>>, VKMLEngineError> {
        match self {
            TensorData::CPU(cell) => Ok(cell.borrow()),
            TensorData::GPU { .. } => Err(VKMLEngineError::VulkanLoadError(
                "Expected CPU tensor, found GPU tensor".to_string(),
            )),
            TensorData::Unallocated => Err(VKMLEngineError::VulkanLoadError(
                "Cannot borrow data from unallocated tensor".to_string(),
            )),
        }
    }

    pub fn borrow_mut_cpu_data(&self) -> Result<RefMut<Vec<f32>>, VKMLEngineError> {
        match self {
            TensorData::CPU(cell) => Ok(cell.borrow_mut()),
            TensorData::GPU { .. } => Err(VKMLEngineError::VulkanLoadError(
                "Expected CPU tensor, found GPU tensor".to_string(),
            )),
            TensorData::Unallocated => Err(VKMLEngineError::VulkanLoadError(
                "Cannot borrow data from unallocated tensor".to_string(),
            )),
        }
    }

    pub fn update_data(&self, data: Vec<f32>) -> Result<(), VKMLEngineError> {
        let expected_elements = match self {
            TensorData::CPU(cpu_data) => cpu_data.borrow().len(),
            TensorData::GPU { memory, .. } => (memory.size as usize) / std::mem::size_of::<f32>(),
            TensorData::Unallocated => {
                return Err(VKMLEngineError::VulkanLoadError(
                    "Cannot update unallocated tensor".to_string(),
                ));
            }
        };

        if data.len() != expected_elements {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Input data size mismatch: expected {} elements, got {}",
                expected_elements,
                data.len()
            )));
        }

        match self {
            TensorData::CPU(cpu_data) => {
                *cpu_data.borrow_mut() = data;
                Ok(())
            }
            TensorData::GPU { memory, .. } => memory
                .copy_into(&data)
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string())),
            TensorData::Unallocated => unreachable!(),
        }
    }

    pub fn is_allocated(&self) -> bool {
        !matches!(self, TensorData::Unallocated)
    }

    pub fn location_string(&self) -> String {
        match self {
            TensorData::CPU(_) => "CPU Tensor".to_string(),
            TensorData::GPU { gpu_idx, .. } => format!("GPU {} Tensor", gpu_idx),
            TensorData::Unallocated => "Unallocated Tensor".to_string(),
        }
    }

    pub fn get_gpu_idx(&self) -> Option<usize> {
        match self {
            TensorData::CPU(_) => None,
            TensorData::GPU { gpu_idx, .. } => Some(*gpu_idx),
            TensorData::Unallocated => None,
        }
    }

    pub fn new_cpu(data: Vec<f32>) -> Self {
        TensorData::CPU(RefCell::new(data))
    }

    pub fn new_gpu(gpu_idx: usize, memory: GPUMemory) -> Self {
        TensorData::GPU { gpu_idx, memory }
    }

    pub fn size_in_bytes(&self) -> u64 {
        match self {
            TensorData::CPU(data) => (data.borrow().len() * std::mem::size_of::<f32>()) as u64,
            TensorData::GPU { memory, .. } => memory.size,
            TensorData::Unallocated => 0,
        }
    }
}
