use std::{ffi::CString, ptr, sync::Arc};
use vulkanalia::{
    Device, Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, DeviceV1_0, InstanceV1_0},
};

use crate::{compute::memory_tracker::MemoryTracker, dataloader::error::VKMLError};

use super::{compute_pipelines::ComputePipelines, gpu_memory::GPUMemory, vk_gpu_info::GPUInfo};

// TODO: Performance gains when needing to multiple tasks in sequence
// TODO: Generalise the usage a little bit more
// NOTE: Get it working, then simplify

// TODO: Look at VK_KHR_device_group. I Think we want to stick with individually managed GPUs though

pub struct GPU {
    entry: Arc<Entry>,
    instance: Instance,
    pub device: Device,
    physical_device: vk::PhysicalDevice,
    compute_queues: Vec<vk::Queue>,
    command_pool: vk::CommandPool,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    compute_pipelines: ComputePipelines,
    memory_tracker: MemoryTracker,
}

impl GPU {
    pub fn new(device_index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;
            let entry = Arc::new(
                Entry::new(loader).map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?,
            );
            let aname = CString::new("VK GPU")?;

            let appinfo = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                next: ptr::null(),
                application_name: aname.as_ptr(),
                application_version: vk::make_version(1, 0, 0),
                engine_name: aname.as_ptr(),
                engine_version: vk::make_version(1, 0, 0),
                api_version: vk::make_version(1, 0, 0),
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                next: ptr::null(),
                flags: vk::InstanceCreateFlags::empty(),
                application_info: &appinfo,
                enabled_layer_count: 0,
                enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                enabled_extension_names: ptr::null(),
            };

            let instance = entry.create_instance(&create_info, None)?;

            let physical_devices = instance.enumerate_physical_devices()?;
            let physical_device = *physical_devices
                .get(device_index)
                .ok_or("GPU index out of range")?;

            let queue_family_index = instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .ok_or("No compute queue family found")?;

            // Get queue family properties to find the number of available queues
            let queue_family_props =
                instance.get_physical_device_queue_family_properties(physical_device);

            let queue_count = queue_family_props[queue_family_index as usize].queue_count;

            // Create queue info with priorities for all queues
            let queue_priorities = vec![1.0; queue_count as usize];

            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count,
                queue_priorities: queue_priorities.as_ptr(),
            };

            let device_features = vk::PhysicalDeviceFeatures::default();

            // Create device with requested queue count
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: 1,
                queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                enabled_layer_names: std::ptr::null(),
                enabled_extension_count: 0,
                enabled_extension_names: ptr::null(),
                enabled_features: &device_features,
            };

            let device = instance.create_device(physical_device, &device_create_info, None)?;

            // Get all compute queues
            let mut compute_queues = Vec::with_capacity(queue_count as usize);
            for i in 0..queue_count {
                compute_queues.push(device.get_device_queue(queue_family_index, i));
            }

            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                next: ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
            };

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            let bindings = [
                // Input buffer 1 (used by all operations)
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                },
                // Input buffer 2 (used by binary operations like Add, Mul, etc.)
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                },
                // Output buffer (binding used by most operations)
                vk::DescriptorSetLayoutBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                },
                // Additional bindings if needed (bias buffer for Conv2D, etc.)
                vk::DescriptorSetLayoutBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                },
            ];

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                binding_count: bindings.len() as u32,
                bindings: bindings.as_ptr(),
            };

            let descriptor_set_layout =
                device.create_descriptor_set_layout(&descriptor_layout_info, None)?;

            let pool_sizes = [vk::DescriptorPoolSize {
                type_: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1000, // TODO: check how high this actually needs to be
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                next: ptr::null(),
                flags: vk::DescriptorPoolCreateFlags::empty(),
                max_sets: 500, // TODO: check how high this actually needs to be
                pool_size_count: pool_sizes.len() as u32,
                pool_sizes: pool_sizes.as_ptr(),
            };

            let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

            let compute_pipelines = ComputePipelines::new(&device, descriptor_set_layout)?;

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let total_memory = {
                let device_local_heap_index = (0..memory_properties.memory_type_count)
                    .find(|&i| {
                        let memory_type = memory_properties.memory_types[i as usize];
                        memory_type
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                    .map(|i| memory_properties.memory_types[i as usize].heap_index)
                    .unwrap_or(0);

                memory_properties.memory_heaps[device_local_heap_index as usize].size
            };

            // TODO: Attempt to figure out why intel iGPU layer shape limit is 8000*8000
            // Set limit as property of the gpu

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                compute_queues,
                command_pool,
                queue_family_index,
                descriptor_pool,
                descriptor_set_layout,
                compute_pipelines,
                memory_tracker: MemoryTracker::new((total_memory as f64 * 0.6) as u64), // TODO: 60%, Kept for for testing at the moment
            })
        }
    }

    unsafe fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
        let aname = CString::new("gpu_mm")?;
        let appinfo = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            next: ptr::null(),
            application_name: aname.as_ptr(),
            application_version: vk::make_version(1, 0, 0),
            engine_name: aname.as_ptr(),
            engine_version: vk::make_version(1, 0, 0),
            api_version: vk::make_version(1, 0, 0),
        };

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            application_info: &appinfo,
            enabled_layer_count: 0,
            enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            enabled_extension_names: ptr::null(),
        };

        Ok(unsafe { entry.create_instance(&create_info, None) }?)
    }

    pub fn total_memory(&self) -> u64 {
        unsafe {
            let memory_properties = self
                .instance
                .get_physical_device_memory_properties(self.physical_device);

            let device_local_heap_index = (0..memory_properties.memory_type_count)
                .find(|&i| {
                    let memory_type = memory_properties.memory_types[i as usize];
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .map(|i| memory_properties.memory_types[i as usize].heap_index)
                .unwrap_or(0);

            memory_properties.memory_heaps[device_local_heap_index as usize].size
        }
    }

    pub fn available_gpus() -> Result<Vec<GPUInfo>, VKMLError> {
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;
            let entry =
                Entry::new(loader).map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            let instance = Self::create_instance(&entry)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            let physical_devices = instance
                .enumerate_physical_devices()
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            // Create GPUInfo for each device and filter for compute support
            let mut gpu_infos: Vec<_> = physical_devices
                .iter()
                .enumerate()
                .map(|(idx, &device)| GPUInfo::new(&instance, device, idx))
                .filter(|info| info.has_compute)
                .collect();

            // Sort GPUs: discrete first, then by memory size
            gpu_infos.sort_by_key(|gpu| {
                (
                    gpu.device_type != vk::PhysicalDeviceType::DISCRETE_GPU,
                    std::cmp::Reverse(gpu.total_memory),
                )
            });

            instance.destroy_instance(None);
            Ok(gpu_infos)
        }
    }

    pub fn move_to_gpu_as_f32(
        &self,
        data: &[f32],
    ) -> Result<GPUMemory, Box<dyn std::error::Error>> {
        let size_in_bytes = (data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;

        unsafe {
            // Create buffer for f32 data
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: ptr::null(),
            };

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);

            let memory_type = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            // Map memory and write data
            let data_ptr =
                self.device
                    .map_memory(memory, 0, size_in_bytes, vk::MemoryMapFlags::empty())?
                    as *mut f32;

            // Copy the data
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());

            self.device.unmap_memory(memory);

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_info = vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: size_in_bytes,
            };

            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_info,
                image_info: ptr::null(),
                texel_buffer_view: ptr::null(),
            };

            self.device
                .update_descriptor_sets(&[write_descriptor_set], &[] as &[vk::CopyDescriptorSet]);

            Ok(GPUMemory {
                buffer,
                memory,
                size: size_in_bytes,
                device: Arc::new(self.device.clone()),
                descriptor_set,
            })
        }
    }

    pub fn allocate_uninitialised_gpu_memory_f32(
        &self,
        num_elements: usize,
    ) -> Result<GPUMemory, Box<dyn std::error::Error>> {
        let size_in_bytes = (num_elements * std::mem::size_of::<f32>()) as vk::DeviceSize;

        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: ptr::null(),
            };

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);

            let memory_type = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            // dont initialise the memory, it will be filled. eg, GPU weight init shaders

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_info = vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: size_in_bytes,
            };

            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_info,
                image_info: ptr::null(),
                texel_buffer_view: ptr::null(),
            };

            self.device
                .update_descriptor_sets(&[write_descriptor_set], &[] as &[vk::CopyDescriptorSet]);

            Ok(GPUMemory {
                buffer,
                memory,
                size: size_in_bytes,
                device: Arc::new(self.device.clone()),
                descriptor_set,
            })
        }
    }

    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        unsafe {
            let mem_properties = self
                .instance
                .get_physical_device_memory_properties(self.physical_device);

            for i in 0..mem_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && mem_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return i;
                }
            }

            panic!("Failed to find suitable memory type")
        }
    }

    pub fn allocate_memory(&self, size: u64) {
        self.memory_tracker.allocate(size)
    }

    pub fn deallocate_memory(&self, size: u64) {
        self.memory_tracker.deallocate(size)
    }

    pub fn available_memory(&self) -> u64 {
        self.memory_tracker.get_available()
    }

    pub fn submit_command_buffers_and_wait(
        &self,
        command_buffers: &[vk::CommandBuffer],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if command_buffers.is_empty() {
            return Ok(());
        }

        unsafe {
            // Split command buffers evenly across queues
            // No idea what happens per device vendor implementation
            // I assume hardware schedulers still only operate one queue at a time
            // But, is the queue maximum equal across all queues in total, or can one queue use it all if the others are empty?
            // Also what if a device can benefit from using the multiple queues in hardware?
            // Is it better for sequential operations to be in the same queue?
            let buffers_per_queue =
                (command_buffers.len() + self.compute_queues.len() - 1) / self.compute_queues.len();
            let mut fences = Vec::with_capacity(self.compute_queues.len());

            // For each queue, create a batch of command buffers
            for (queue_idx, chunk) in command_buffers.chunks(buffers_per_queue).enumerate() {
                if chunk.is_empty() {
                    continue;
                }

                // Create a fence for this submission to track completion
                let fence_info = vk::FenceCreateInfo {
                    s_type: vk::StructureType::FENCE_CREATE_INFO,
                    next: std::ptr::null(),
                    flags: vk::FenceCreateFlags::empty(),
                };

                let fence = self.device.create_fence(&fence_info, None)?;
                fences.push(fence);

                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    wait_semaphores: std::ptr::null(),
                    wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: chunk.len() as u32,
                    command_buffers: chunk.as_ptr(),
                    signal_semaphore_count: 0,
                    signal_semaphores: std::ptr::null(),
                };

                // Get queue index, wrap around if needed
                let queue_idx = queue_idx % self.compute_queues.len();

                self.device
                    .queue_submit(self.compute_queues[queue_idx], &[submit_info], fence)?;
            }

            // Wait for all fences to signal completion
            if !fences.is_empty() {
                self.device.wait_for_fences(&fences, true, std::u64::MAX)?;

                for fence in fences {
                    self.device.destroy_fence(fence, None);
                }
            }

            Ok(())
        }
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_compute_pipelines(&self) -> &ComputePipelines {
        &self.compute_pipelines
    }

    pub fn get_descriptor_pool(&self) -> &vk::DescriptorPool {
        &self.descriptor_pool
    }

    pub fn get_descriptor_set_layout(&self) -> &vk::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn get_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for GPU {
    fn drop(&mut self) {
        unsafe {
            self.compute_pipelines.cleanup(&self.device);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
