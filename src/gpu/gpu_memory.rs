use std::sync::Arc;

use vulkanalia::{Device, vk, vk::DeviceV1_0};

use crate::error::VKMLError;

pub struct GPUMemory {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub device: Arc<Device>,
}

impl GPUMemory {
    pub fn new(
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
        device: Arc<Device>,
    ) -> Self {
        Self {
            buffer,
            memory,
            size,
            device,
        }
    }

    /// Copy raw bytes into GPU memory.
    pub fn copy_into(&self, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let data_size = data.len() as vk::DeviceSize;

        if data_size > self.size {
            return Err(format!(
                "Data size {} exceeds GPU buffer size {}",
                data_size, self.size
            )
            .into());
        }

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(self.memory, 0, data_size, vk::MemoryMapFlags::empty())?
                    as *mut u8;

            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());

            self.device.unmap_memory(self.memory);
        }

        Ok(())
    }

    /// Read raw bytes from GPU memory.
    pub fn read_memory(&self) -> Result<Vec<u8>, VKMLError> {
        let mut output_data = vec![0u8; self.size as usize];

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?
                    as *const u8;

            std::ptr::copy_nonoverlapping(data_ptr, output_data.as_mut_ptr(), output_data.len());

            self.device.unmap_memory(self.memory);
        }

        Ok(output_data)
    }
}
