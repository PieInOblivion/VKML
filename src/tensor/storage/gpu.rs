use super::r#trait::TensorStorageOps;
use crate::{gpu::gpu_memory::GPUMemory, utils::expect_msg::ExpectMsg};
use std::ops::{Deref, DerefMut};

pub struct GpuTensorStorage {
    gpu_idx: usize,
    memory: GPUMemory,
}

impl GpuTensorStorage {
    pub fn new(gpu_idx: usize, memory: GPUMemory) -> Self {
        Self { gpu_idx, memory }
    }

    /// Get direct access to GPU memory (for Vulkan operations)
    pub fn memory(&self) -> &GPUMemory {
        &self.memory
    }
}

impl TensorStorageOps for GpuTensorStorage {
    type ReadGuard<'a> = GpuReadGuard;
    type WriteGuard<'a> = GpuWriteGuard<'a>;

    fn read_data(&self) -> Self::ReadGuard<'_> {
        let data = self
            .memory
            .read_memory()
            .expect_msg("Failed to read GPU memory");
        GpuReadGuard { data }
    }

    fn write_data(&self) -> Self::WriteGuard<'_> {
        let data = self
            .memory
            .read_memory()
            .expect_msg("Failed to read GPU memory");
        GpuWriteGuard { data, memory: &self.memory }
    }

    fn get_data(&self) -> Vec<u8> {
        self.memory
            .read_memory()
            .expect_msg("Failed to read GPU memory")
    }

    fn update_data(&self, data: Vec<u8>) {
        let expected_bytes = self.memory.size as usize;
        if data.len() != expected_bytes {
            panic!(
                "Input data size mismatch: expected {} bytes, got {}",
                expected_bytes,
                data.len()
            );
        }
        self.memory
            .copy_into(&data)
            .expect_msg("Failed to copy data to GPU memory");
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

// Read guard for GPU - holds the copied bytes
pub struct GpuReadGuard {
    data: Vec<u8>,
}

impl Deref for GpuReadGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// Write guard for GPU - copies back to GPU on drop
pub struct GpuWriteGuard<'a> {
    data: Vec<u8>,
    memory: &'a GPUMemory,
}

impl<'a> Deref for GpuWriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> DerefMut for GpuWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a> Drop for GpuWriteGuard<'a> {
    fn drop(&mut self) {
        self.memory
            .copy_into(&self.data)
            .expect_msg("Failed to write data back to GPU memory");
    }
}
