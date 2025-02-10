use std::sync::Arc;

use ash::{vk, Device};

pub struct GPUMemory {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub device: Arc<Device>,
    pub descriptor_set: vk::DescriptorSet,
}

impl GPUMemory {
    pub fn new(buffer: vk::Buffer, memory: vk::DeviceMemory, size: vk::DeviceSize, device: Arc<Device>, descriptor_set: vk::DescriptorSet) -> Self {
        Self {
            buffer,
            memory,
            size,
            device,
            descriptor_set
        }
    }

    pub fn copy_into(&self, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let data_size = (data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;

        if data_size > self.size {
            return Err("Source data is larger than destination GPU memory".into());
        }

        unsafe {
            let data_ptr = self.device.map_memory(
                self.memory,
                0,
                data_size,
                vk::MemoryMapFlags::empty()
            )? as *mut f32;

            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                data_ptr,
                data.len()
            );
        }

        Ok(())
    }

    pub fn read_memory(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let num_floats = (self.size as usize) / std::mem::size_of::<f32>();
        let mut output_data = vec![0f32; num_floats];

        unsafe {
            let data_ptr = self.device.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty()
            )? as *const f32;

            std::ptr::copy_nonoverlapping(
                data_ptr,
                output_data.as_mut_ptr(),
                num_floats
            );
        }

        Ok(output_data)
    }
}