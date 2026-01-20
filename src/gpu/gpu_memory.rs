use std::sync::{Arc, Weak};

use vulkanalia::{vk, vk::DeviceV1_0};

use crate::error::VKMLError;

use super::vk_gpu::Gpu;

pub struct GPUMemory {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub properties: vk::MemoryPropertyFlags,
    gpu: Weak<Gpu>,
}

impl GPUMemory {
    pub fn new(
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
        properties: vk::MemoryPropertyFlags,
        gpu: &Arc<Gpu>,
    ) -> Self {
        Self {
            buffer,
            memory,
            size,
            properties,
            gpu: Arc::downgrade(gpu),
        }
    }

    fn upgrade_gpu(&self) -> Result<Arc<Gpu>, VKMLError> {
        self.gpu.upgrade().ok_or_else(|| {
            VKMLError::Vulkan("GPU allocation reference dropped before use".to_string())
        })
    }

    /// Copy raw bytes into GPU memory. Falls back to staging when not host-visible.
    pub fn copy_into(&self, data: &[u8]) -> Result<(), VKMLError> {
        let data_size = data.len() as vk::DeviceSize;

        if data_size > self.size {
            return Err(VKMLError::Vulkan(format!(
                "Data size {} exceeds GPU buffer size {}",
                data_size, self.size
            )));
        }

        if self
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            let gpu = self.upgrade_gpu()?;
            unsafe {
                let data_ptr = gpu.get_device().map_memory(
                    self.memory,
                    0,
                    data_size,
                    vk::MemoryMapFlags::empty(),
                )? as *mut u8;

                std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());

                gpu.get_device().unmap_memory(self.memory);
            }
            Ok(())
        } else {
            let gpu = self.upgrade_gpu()?;
            gpu.write_through_staging(self, data)
        }
    }

    /// Read raw bytes from GPU memory. Falls back to staging when not host-visible.
    pub fn read_memory(&self) -> Result<Box<[u8]>, VKMLError> {
        if self
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            let gpu = self.upgrade_gpu()?;
            let mut buffer = Box::new_uninit_slice(self.size as usize);

            unsafe {
                let data_ptr = gpu.get_device().map_memory(
                    self.memory,
                    0,
                    self.size,
                    vk::MemoryMapFlags::empty(),
                )? as *const u8;

                let buffer_ptr = buffer.as_mut_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(data_ptr, buffer_ptr, buffer.len());

                gpu.get_device().unmap_memory(self.memory);
            }

            let output = unsafe { buffer.assume_init() };

            Ok(output)
        } else {
            let gpu = self.upgrade_gpu()?;
            gpu.read_through_staging(self)
        }
    }
}
