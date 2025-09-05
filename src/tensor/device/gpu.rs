use std::any::Any;

use crate::{gpu::gpu_memory::GPUMemory, tensor::data::TensorData};

/// Simple GPU buffer wrapper around GPUMemory (owns the GPU allocation handle)
pub struct GpuData {
    pub memory: GPUMemory,
}

impl TensorData for GpuData {
    fn len_bytes(&self) -> usize {
        self.memory.size as usize
    }

    fn read(&self) -> Vec<u8> {
        self.memory
            .read_memory()
            .expect("Failed to read GPU memory in read_host")
    }

    fn write(&mut self, data: &[u8]) {
        let expected = self.memory.size as usize;
        assert_eq!(data.len(), expected);
        self.memory
            .copy_into(data)
            .expect("Failed to copy data into GPU memory");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
