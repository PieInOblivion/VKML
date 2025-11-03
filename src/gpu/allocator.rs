use std::cmp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use vulkanalia::vk::{self, DeviceV1_0};

use crate::error::VKMLError;
use crate::utils::expect_msg::ExpectMsg;

use super::gpu_memory::GPUMemory;
use super::vk_gpu::Gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAccessMode {
    /// All tensors are host visible; staging is unnecessary.
    DirectAllHostVisible,
    /// Tensors remain device local and host access occurs through staging.
    DeviceLocalWithStaging,
}

pub struct GpuAllocator {
    host_visible_reserved: AtomicU64,
    host_access_mode: RwLock<HostAccessMode>,
    staging_buffer: OnceLock<Mutex<GPUMemory>>,
}

impl GpuAllocator {
    pub fn new() -> Self {
        Self {
            host_visible_reserved: AtomicU64::new(0),
            host_access_mode: RwLock::new(HostAccessMode::DeviceLocalWithStaging),
            staging_buffer: OnceLock::new(),
        }
    }

    pub fn host_access_mode(&self) -> HostAccessMode {
        *self.host_access_mode.read().unwrap()
    }

    pub fn set_host_access_mode(&self, mode: HostAccessMode) {
        *self.host_access_mode.write().unwrap() = mode;
    }

    pub fn direct_host_mode(&self) -> bool {
        matches!(
            self.host_access_mode(),
            HostAccessMode::DirectAllHostVisible
        )
    }

    pub fn set_host_visible_reserved(&self, bytes: u64) {
        self.host_visible_reserved.store(bytes, Ordering::Relaxed);
    }

    pub fn host_visible_reserved(&self) -> u64 {
        self.host_visible_reserved.load(Ordering::Relaxed)
    }

    pub fn staging_buffer_info(&self) -> Option<(vk::DeviceSize, vk::MemoryPropertyFlags)> {
        self.staging_buffer
            .get()
            .and_then(|mutex| match mutex.lock() {
                Ok(buffer) => Some((buffer.size, buffer.properties)),
                Err(_) => None,
            })
    }

    pub fn preview(&self, gpu: &Gpu, reserved_host_visible_bytes: u64) -> (usize, bool) {
        let total_memory = gpu.memory_total().max(1);
        let mut size_bytes = (total_memory / 20).max(1); // target: 5% of total memory
        let mut device_local = false;

        let host_visible_budget = gpu.host_visible_device_local_bytes();
        if host_visible_budget > reserved_host_visible_bytes {
            let available = host_visible_budget - reserved_host_visible_bytes;
            let visibility_threshold = (total_memory / 100).max(1); // require ~1% before preferring device-local
            if available >= visibility_threshold {
                size_bytes = available.min(size_bytes);
                device_local = true;
            }
        }

        if size_bytes == 0 {
            size_bytes = 1;
        }

        (size_bytes as usize, device_local)
    }

    fn plan_current(&self, gpu: &Gpu) -> (usize, bool) {
        self.preview(gpu, self.host_visible_reserved())
    }

    pub fn get_or_create_staging_buffer<'a>(&'a self, gpu: &Arc<Gpu>) -> &'a Mutex<GPUMemory> {
        debug_assert!(
            !self.direct_host_mode(),
            "Staging buffer requested while GPU is in direct host access mode",
        );

        self.staging_buffer.get_or_init(|| {
            let gpu = Arc::clone(gpu);

            unsafe {
                let (staging_size, device_local_staging) = self.plan_current(gpu.as_ref());

                let buffer_info = vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    next: std::ptr::null(),
                    flags: vk::BufferCreateFlags::empty(),
                    size: staging_size as vk::DeviceSize,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_family_index_count: 0,
                    queue_family_indices: std::ptr::null(),
                };

                let buffer = gpu
                    .get_device()
                    .create_buffer(&buffer_info, None)
                    .expect_msg("Failed to create staging buffer");
                let mem_requirements = gpu.get_device().get_buffer_memory_requirements(buffer);

                let requested_properties = if device_local_staging {
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::DEVICE_LOCAL
                } else {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                };

                let memory_type =
                    gpu.find_memory_type(mem_requirements.memory_type_bits, requested_properties);

                let alloc_info = vk::MemoryAllocateInfo {
                    s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                    next: std::ptr::null(),
                    allocation_size: mem_requirements.size,
                    memory_type_index: memory_type,
                };

                let memory = gpu
                    .get_device()
                    .allocate_memory(&alloc_info, None)
                    .expect_msg("Failed to allocate staging memory");
                gpu.get_device()
                    .bind_buffer_memory(buffer, memory, 0)
                    .expect_msg("Failed to bind staging buffer memory");

                let staging = GPUMemory::new(
                    buffer,
                    memory,
                    staging_size as vk::DeviceSize,
                    requested_properties,
                    &gpu,
                );

                Mutex::new(staging)
            }
        })
    }

    pub fn move_to_gpu_host_visible(
        &self,
        gpu: &Arc<Gpu>,
        bytes: &[u8],
    ) -> Result<GPUMemory, VKMLError> {
        if !self.direct_host_mode() {
            return self.move_to_gpu_host_not_visible(gpu, bytes);
        }

        let size_in_bytes = bytes.len() as vk::DeviceSize;
        gpu.memory_allocate_usage(size_in_bytes);

        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };

            let buffer = gpu.get_device().create_buffer(&buffer_info, None)?;
            let mem_requirements = gpu.get_device().get_buffer_memory_requirements(buffer);

            let memory_type = gpu.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: std::ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = gpu.get_device().allocate_memory(&alloc_info, None)?;
            gpu.get_device().bind_buffer_memory(buffer, memory, 0)?;

            let data_ptr = gpu.get_device().map_memory(
                memory,
                0,
                size_in_bytes,
                vk::MemoryMapFlags::empty(),
            )? as *mut u8;

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());

            gpu.get_device().unmap_memory(memory);

            let properties = vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL;

            Ok(GPUMemory::new(
                buffer,
                memory,
                size_in_bytes,
                properties,
                gpu,
            ))
        }
    }

    pub fn move_to_gpu_host_not_visible(
        &self,
        gpu: &Arc<Gpu>,
        bytes: &[u8],
    ) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes.len() as vk::DeviceSize;
        gpu.memory_allocate_usage(size_in_bytes);

        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };

            let buffer = gpu.get_device().create_buffer(&buffer_info, None)?;
            let mem_requirements = gpu.get_device().get_buffer_memory_requirements(buffer);

            let memory_type = gpu.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: std::ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = gpu.get_device().allocate_memory(&alloc_info, None)?;
            gpu.get_device().bind_buffer_memory(buffer, memory, 0)?;

            let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let dest = GPUMemory::new(buffer, memory, size_in_bytes, properties, gpu);

            self.write_through_staging(gpu, &dest, bytes)?;

            Ok(dest)
        }
    }

    pub fn allocate_uninitialised_gpu_memory(
        &self,
        gpu: &Arc<Gpu>,
        bytes: usize,
        host_visible: bool,
    ) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes as vk::DeviceSize;
        gpu.memory_allocate_usage(size_in_bytes);

        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };

            let buffer = gpu.get_device().create_buffer(&buffer_info, None)?;
            let mem_requirements = gpu.get_device().get_buffer_memory_requirements(buffer);

            let use_host_visible = host_visible && self.direct_host_mode();

            let properties = if use_host_visible {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };

            let memory_type = gpu.find_memory_type(mem_requirements.memory_type_bits, properties);

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: std::ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = gpu.get_device().allocate_memory(&alloc_info, None)?;
            gpu.get_device().bind_buffer_memory(buffer, memory, 0)?;

            Ok(GPUMemory::new(
                buffer,
                memory,
                size_in_bytes,
                properties,
                gpu,
            ))
        }
    }

    pub fn write_through_staging(
        &self,
        gpu: &Arc<Gpu>,
        dest: &GPUMemory,
        data: &[u8],
    ) -> Result<(), VKMLError> {
        debug_assert!(
            !self.direct_host_mode(),
            "write_through_staging should not be used in direct host mode",
        );

        if data.len() as vk::DeviceSize > dest.size {
            return Err(VKMLError::Vulkan(format!(
                "Attempted to write {} bytes into buffer sized {}",
                data.len(),
                dest.size
            )));
        }

        let staging_mutex = self.get_or_create_staging_buffer(gpu);
        let staging_guard = staging_mutex.lock().unwrap();
        let staging_size = staging_guard.size as usize;

        if staging_size == 0 {
            return Err(VKMLError::Vulkan(
                "Staging buffer must be at least 1 byte".to_string(),
            ));
        }

        let command_buffer_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            next: std::ptr::null(),
            command_pool: gpu.get_command_pool(),
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            next: std::ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };

        unsafe {
            let command_buffers = gpu
                .get_device()
                .allocate_command_buffers(&command_buffer_info)?;
            let command_buffer = *command_buffers
                .first()
                .ok_or_else(|| VKMLError::Vulkan("Failed to allocate command buffer".into()))?;

            let fence = gpu.get_device().create_fence(&fence_info, None)?;

            let result = (|| -> Result<(), VKMLError> {
                let mut offset = 0usize;
                while offset < data.len() {
                    let remaining = data.len() - offset;
                    let chunk_size = cmp::min(staging_size, remaining);
                    let chunk = &data[offset..offset + chunk_size];

                    staging_guard.copy_into(chunk)?;

                    gpu.begin_command_buffer(command_buffer)?;

                    let copy_region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset as vk::DeviceSize,
                        size: chunk_size as vk::DeviceSize,
                    };

                    gpu.get_device().cmd_copy_buffer(
                        command_buffer,
                        staging_guard.buffer,
                        dest.buffer,
                        &[copy_region],
                    );

                    gpu.end_command_buffer(command_buffer)?;

                    gpu.submit_with_fence(&[command_buffer], Some(fence))?;
                    gpu.wait_and_reset_fence(fence)?;

                    gpu.get_device().reset_command_buffer(
                        command_buffer,
                        vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                    )?;

                    offset += chunk_size;
                }
                Ok(())
            })();

            gpu.get_device()
                .free_command_buffers(gpu.get_command_pool(), &[command_buffer]);
            gpu.get_device().destroy_fence(fence, None);

            result
        }
    }

    pub fn read_through_staging(
        &self,
        gpu: &Arc<Gpu>,
        source: &GPUMemory,
    ) -> Result<Vec<u8>, VKMLError> {
        debug_assert!(
            !self.direct_host_mode(),
            "read_through_staging should not be used in direct host mode",
        );

        let total_bytes = source.size as usize;
        let mut output = vec![0u8; total_bytes];

        let staging_mutex = self.get_or_create_staging_buffer(gpu);
        let staging_guard = staging_mutex.lock().unwrap();
        let staging_size = staging_guard.size as usize;

        if staging_size == 0 {
            return Err(VKMLError::Vulkan(
                "Staging buffer must be at least 1 byte".to_string(),
            ));
        }

        let command_buffer_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            next: std::ptr::null(),
            command_pool: gpu.get_command_pool(),
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            next: std::ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };

        unsafe {
            let command_buffers = gpu
                .get_device()
                .allocate_command_buffers(&command_buffer_info)?;
            let command_buffer = *command_buffers
                .first()
                .ok_or_else(|| VKMLError::Vulkan("Failed to allocate command buffer".into()))?;

            let fence = gpu.get_device().create_fence(&fence_info, None)?;

            let result = (|| -> Result<(), VKMLError> {
                let mut offset = 0usize;
                while offset < total_bytes {
                    let remaining = total_bytes - offset;
                    let chunk_size = cmp::min(staging_size, remaining);

                    gpu.begin_command_buffer(command_buffer)?;

                    let copy_region = vk::BufferCopy {
                        src_offset: offset as vk::DeviceSize,
                        dst_offset: 0,
                        size: chunk_size as vk::DeviceSize,
                    };

                    gpu.get_device().cmd_copy_buffer(
                        command_buffer,
                        source.buffer,
                        staging_guard.buffer,
                        &[copy_region],
                    );

                    gpu.end_command_buffer(command_buffer)?;

                    gpu.submit_with_fence(&[command_buffer], Some(fence))?;
                    gpu.wait_and_reset_fence(fence)?;

                    let data_ptr = gpu.get_device().map_memory(
                        staging_guard.memory,
                        0,
                        chunk_size as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )? as *const u8;

                    let dst_slice = &mut output[offset..offset + chunk_size];
                    std::ptr::copy_nonoverlapping(data_ptr, dst_slice.as_mut_ptr(), chunk_size);
                    gpu.get_device().unmap_memory(staging_guard.memory);

                    gpu.get_device().reset_command_buffer(
                        command_buffer,
                        vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                    )?;

                    offset += chunk_size;
                }

                Ok(())
            })();

            gpu.get_device()
                .free_command_buffers(gpu.get_command_pool(), &[command_buffer]);
            gpu.get_device().destroy_fence(fence, None);

            result?;
        }

        Ok(output)
    }
}
