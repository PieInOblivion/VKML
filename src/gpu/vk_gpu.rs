use ash::{Device, Entry, Instance, vk};
use std::{ffi::CString, ptr, sync::Arc};

use crate::{
    compute::memory_tracker::MemoryTracker,
    dataloader::error::VKMLEngineError,
    tensor::{compute_tensor::ComputeTensor, tensor_data::TensorData},
};

use super::{
    compute_pipelines::{ComputePipelines, GPUMemoryOperation},
    gpu_memory::GPUMemory,
    vk_gpu_info::GPUInfo,
};

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
    pub command_pool: vk::CommandPool,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    compute_pipelines: ComputePipelines,
    memory_tracker: MemoryTracker,
}

impl GPU {
    pub fn new(device_index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let entry = Arc::new(Entry::load()?);
            let app_name = CString::new("VK GPU")?;

            let appinfo = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 1, 0, 0),
                p_engine_name: app_name.as_ptr(),
                engine_version: vk::make_api_version(0, 1, 0, 0),
                api_version: vk::make_api_version(0, 1, 0, 0),
                _marker: std::marker::PhantomData,
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::InstanceCreateFlags::empty(),
                p_application_info: &appinfo,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
                _marker: std::marker::PhantomData,
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
                p_next: std::ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count,
                p_queue_priorities: queue_priorities.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let device_features = vk::PhysicalDeviceFeatures::default();

            // Create device with requested queue count
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                pp_enabled_layer_names: std::ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: std::ptr::null(),
                p_enabled_features: &device_features,
                _marker: std::marker::PhantomData,
            };

            let device = instance.create_device(physical_device, &device_create_info, None)?;

            // Get all compute queues
            let mut compute_queues = Vec::with_capacity(queue_count as usize);
            for i in 0..queue_count {
                compute_queues.push(device.get_device_queue(queue_family_index, i));
            }

            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
                _marker: std::marker::PhantomData,
            };

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            let bindings = [
                // Input buffer 1 (used by all operations)
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Input buffer 2 (used by binary operations like Add, Mul, etc.)
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Output buffer (binding used by most operations)
                vk::DescriptorSetLayoutBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Additional bindings if needed (bias buffer for Conv2D, etc.)
                vk::DescriptorSetLayoutBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set_layout =
                device.create_descriptor_set_layout(&descriptor_layout_info, None)?;

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1000, // TODO: check how high this actually needs to be
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DescriptorPoolCreateFlags::empty(),
                max_sets: 500, // TODO: check how high this actually needs to be
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                _marker: std::marker::PhantomData,
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
                memory_tracker: MemoryTracker::new((total_memory as f64 * 0.6) as u64),
            })
        }
    }

    unsafe fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
        let app_name = CString::new("gpu_mm")?;
        let appinfo = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
            _marker: std::marker::PhantomData,
        };

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &appinfo,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            _marker: std::marker::PhantomData,
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

    pub fn available_gpus() -> Result<Vec<GPUInfo>, VKMLEngineError> {
        unsafe {
            let entry =
                Entry::load().map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;

            let instance = Self::create_instance(&entry)
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;

            let physical_devices = instance
                .enumerate_physical_devices()
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;

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
                p_next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);

            let memory_type = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
                _marker: std::marker::PhantomData,
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
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_info = vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: size_in_bytes,
            };

            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info,
                p_image_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .update_descriptor_sets(&[write_descriptor_set], &[]);

            Ok(GPUMemory {
                buffer,
                memory,
                size: size_in_bytes,
                device: Arc::new(self.device.clone()),
                descriptor_set,
            })
        }
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
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
                    return Ok(i);
                }
            }

            Err("Failed to find suitable memory type".into())
        }
    }

    pub fn allocate_memory(&self, size: u64) -> Result<(), VKMLEngineError> {
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
            // But, is the queue maximum equal across all queues in total, or can one queue use it all if the oterhs are empty?
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
                    p_next: std::ptr::null(),
                    flags: vk::FenceCreateFlags::empty(),
                    _marker: std::marker::PhantomData,
                };

                let fence = self.device.create_fence(&fence_info, None)?;
                fences.push(fence);

                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    p_next: std::ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: std::ptr::null(),
                    p_wait_dst_stage_mask: std::ptr::null(),
                    command_buffer_count: chunk.len() as u32,
                    p_command_buffers: chunk.as_ptr(),
                    signal_semaphore_count: 0,
                    p_signal_semaphores: std::ptr::null(),
                    _marker: std::marker::PhantomData,
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

    pub fn create_add_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Add,
        )
    }

    pub fn create_sub_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Subtract,
        )
    }

    pub fn create_mul_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Multiply,
        )
    }

    pub fn create_div_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Divide,
        )
    }

    pub fn create_max_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Maximum,
        )
    }

    pub fn create_min_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_element_wise_command_buffer(
            command_buffer,
            src1,
            src2,
            dst,
            GPUMemoryOperation::Minimum,
        )
    }

    fn create_element_wise_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1: &GPUMemory,
        src2: &GPUMemory,
        dst: &GPUMemory,
        operation: GPUMemoryOperation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src1 buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src1.buffer,
                    offset: 0,
                    range: src1.size,
                },
                // src2 buffer (binding 1)
                vk::DescriptorBufferInfo {
                    buffer: src2.buffer,
                    offset: 0,
                    range: src2.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst.buffer,
                    offset: 0,
                    range: dst.size,
                },
            ];

            let write_descriptor_sets = [
                // src1 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // src2 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[2],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(operation)
                .ok_or(format!("{:?} pipeline not found", operation))?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let workgroup_size = 256;
            let num_elements = dst.size / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size as u64 - 1) / workgroup_size as u64;

            self.device
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_relu_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_unary_operation_command_buffer(
            command_buffer,
            src,
            dst,
            GPUMemoryOperation::ReLU,
        )
    }

    pub fn create_leaky_relu_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
        alpha: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src.buffer,
                    offset: 0,
                    range: src.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst.buffer,
                    offset: 0,
                    range: dst.size,
                },
            ];

            let write_descriptor_sets = [
                // src buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(GPUMemoryOperation::LeakyReLU)
                .ok_or("LeakyReLU pipeline not found")?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            // Push alpha value as a constant
            self.device.cmd_push_constants(
                command_buffer,
                self.compute_pipelines.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &alpha as *const f32 as *const u8,
                    std::mem::size_of::<f32>(),
                ),
            );

            let workgroup_size = 256;
            let num_elements = dst.size / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size as u64 - 1) / workgroup_size as u64;

            self.device
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_sigmoid_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_unary_operation_command_buffer(
            command_buffer,
            src,
            dst,
            GPUMemoryOperation::Sigmoid,
        )
    }

    pub fn create_tanh_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_unary_operation_command_buffer(
            command_buffer,
            src,
            dst,
            GPUMemoryOperation::Tanh,
        )
    }

    pub fn create_gelu_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_unary_operation_command_buffer(
            command_buffer,
            src,
            dst,
            GPUMemoryOperation::GELU,
        )
    }

    pub fn create_silu_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.create_unary_operation_command_buffer(
            command_buffer,
            src,
            dst,
            GPUMemoryOperation::SiLU,
        )
    }

    pub fn create_softmax_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
        dim: usize,
        dims: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Currently we only support softmax on the last dimension
        if dim != dims.len() - 1 {
            return Err(format!("Only softmax on the last dimension is currently implemented, requested dimension: {}", dim).into());
        }

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src.buffer,
                    offset: 0,
                    range: src.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst.buffer,
                    offset: 0,
                    range: dst.size,
                },
            ];

            let write_descriptor_sets = [
                // src buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(GPUMemoryOperation::Softmax)
                .ok_or("Softmax pipeline not found")?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let feature_size = dims[dim];
            let batch_size = src.size as usize / std::mem::size_of::<f32>() / feature_size;

            // Create push constants struct
            #[repr(C)]
            struct SoftmaxPushConstants {
                batch_size: u32,
                feature_size: u32,
            }

            let push_constants = SoftmaxPushConstants {
                batch_size: batch_size as u32,
                feature_size: feature_size as u32,
            };

            // Push constants to the shader
            self.device.cmd_push_constants(
                command_buffer,
                self.compute_pipelines.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const SoftmaxPushConstants as *const u8,
                    std::mem::size_of::<SoftmaxPushConstants>(),
                ),
            );

            // Calculate dispatch size based on batch size
            // One workgroup per batch for now
            let num_workgroups = (batch_size as u64 + 255) / 256;

            self.device
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    // ReLU, Sigmoid, Tanh, GELU, SiLU
    fn create_unary_operation_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
        operation: GPUMemoryOperation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src.buffer,
                    offset: 0,
                    range: src.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst.buffer,
                    offset: 0,
                    range: dst.size,
                },
            ];

            let write_descriptor_sets = [
                // src buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(operation)
                .ok_or(format!("{:?} pipeline not found", operation))?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let workgroup_size = 256;
            let num_elements = dst.size / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size as u64 - 1) / workgroup_size as u64;

            self.device
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_copy_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        dst: &GPUMemory,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            // Copy regions - entire buffer
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src.size,
            };

            self.device
                .cmd_copy_buffer(command_buffer, src.buffer, dst.buffer, &[copy_region]);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_conv2d_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &GPUMemory,
        filters: &GPUMemory,
        bias: Option<&GPUMemory>,
        dst: &GPUMemory,
        src_tensor: &ComputeTensor,
        filters_tensor: &ComputeTensor,
        dst_tensor: &ComputeTensor,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            // Get dimensions from tensor descriptors
            let src_dims = src_tensor.desc.to_dims();
            let filter_dims = filters_tensor.desc.to_dims();
            let dst_dims = dst_tensor.desc.to_dims();

            // Get strides for input, filter, and output tensors
            let src_strides = src_tensor.desc.strides();
            let filter_strides = filters_tensor.desc.strides();
            let dst_strides = dst_tensor.desc.strides();

            // Validate tensor dimensions
            if src_dims.len() != 4 || filter_dims.len() != 4 || dst_dims.len() != 4 {
                return Err("Conv2D requires 4D tensors for input, filters, and output".into());
            }

            let batch_size = src_dims[0];
            let in_channels = src_dims[1];
            let in_height = src_dims[2];
            let in_width = src_dims[3];

            let out_channels = filter_dims[0];
            let filter_in_channels = filter_dims[1];
            let filter_height = filter_dims[2];
            let filter_width = filter_dims[3];

            let out_batch = dst_dims[0];
            let out_channels_check = dst_dims[1];
            let out_height = dst_dims[2];
            let out_width = dst_dims[3];

            // Validation checks
            if batch_size != out_batch {
                return Err(format!(
                    "Batch size mismatch: input={}, output={}",
                    batch_size, out_batch
                )
                .into());
            }

            if out_channels != out_channels_check {
                return Err(format!(
                    "Output channel mismatch: filter={}, output={}",
                    out_channels, out_channels_check
                )
                .into());
            }

            if in_channels != filter_in_channels {
                return Err(format!(
                    "Input channel mismatch: input={}, filter={}",
                    in_channels, filter_in_channels
                )
                .into());
            }

            // Verify output dimensions match expected conv2d output size
            let expected_out_height = (in_height + 2 * padding_h - filter_height) / stride_h + 1;
            let expected_out_width = (in_width + 2 * padding_w - filter_width) / stride_w + 1;

            if out_height != expected_out_height || out_width != expected_out_width {
                return Err(format!(
                    "Output dimensions mismatch. Expected: {}×{}, Got: {}×{}",
                    expected_out_height, expected_out_width, out_height, out_width
                )
                .into());
            }

            // Update descriptor set with input, filter, bias (optional), and output buffers
            let mut buffer_infos = vec![
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src.buffer,
                    offset: 0,
                    range: src.size,
                },
                // filter buffer (binding 1)
                vk::DescriptorBufferInfo {
                    buffer: filters.buffer,
                    offset: 0,
                    range: filters.size,
                },
                // bias buffer (binding 2, optional)
                vk::DescriptorBufferInfo {
                    buffer: bias.map_or(filters.buffer, |b| b.buffer), // Reuse filter buffer if no bias
                    offset: 0,
                    range: bias.map_or(4, |b| b.size), // Min size if no bias
                },
                // dst buffer (binding 3)
                vk::DescriptorBufferInfo {
                    buffer: dst.buffer,
                    offset: 0,
                    range: dst.size,
                },
            ];

            let write_descriptor_sets = vec![
                // Input buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Filter buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Bias buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[2],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // Output buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[3],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(GPUMemoryOperation::Conv2D)
                .ok_or("Conv2D pipeline not found")?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            // Create push constants
            #[repr(C)]
            struct Conv2DPushConstants {
                // Dimensions
                batch_size: u32,
                in_channels: u32,
                in_height: u32,
                in_width: u32,

                filter_out_channels: u32,
                filter_height: u32,
                filter_width: u32,

                out_height: u32,
                out_width: u32,

                // Convolution parameters
                stride_h: u32,
                stride_w: u32,
                padding_h: u32,
                padding_w: u32,

                // Tensor strides (up to 8 values, 4 for each tensor)
                src_stride_0: u32,
                src_stride_1: u32,
                src_stride_2: u32,
                src_stride_3: u32,

                filter_stride_0: u32,
                filter_stride_1: u32,
                filter_stride_2: u32,
                filter_stride_3: u32,

                dst_stride_0: u32,
                dst_stride_1: u32,
                dst_stride_2: u32,
                dst_stride_3: u32,

                use_bias: u32,
            }

            let push_constants = Conv2DPushConstants {
                batch_size: batch_size as u32,
                in_channels: in_channels as u32,
                in_height: in_height as u32,
                in_width: in_width as u32,

                filter_out_channels: out_channels as u32,
                filter_height: filter_height as u32,
                filter_width: filter_width as u32,

                out_height: out_height as u32,
                out_width: out_width as u32,

                stride_h: stride_h as u32,
                stride_w: stride_w as u32,
                padding_h: padding_h as u32,
                padding_w: padding_w as u32,

                // Input tensor strides
                src_stride_0: src_strides[0] as u32,
                src_stride_1: src_strides[1] as u32,
                src_stride_2: src_strides[2] as u32,
                src_stride_3: src_strides[3] as u32,

                // Filter tensor strides
                filter_stride_0: filter_strides[0] as u32,
                filter_stride_1: filter_strides[1] as u32,
                filter_stride_2: filter_strides[2] as u32,
                filter_stride_3: filter_strides[3] as u32,

                // Output tensor strides
                dst_stride_0: dst_strides[0] as u32,
                dst_stride_1: dst_strides[1] as u32,
                dst_stride_2: dst_strides[2] as u32,
                dst_stride_3: dst_strides[3] as u32,

                use_bias: if bias.is_some() { 1 } else { 0 },
            };

            // Push constants to the shader
            self.device.cmd_push_constants(
                command_buffer,
                self.compute_pipelines.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const Conv2DPushConstants as *const u8,
                    std::mem::size_of::<Conv2DPushConstants>(),
                ),
            );

            // Calculate dispatch size based on output dimensions
            // Each thread computes one output element
            let total_output_elements = batch_size * out_channels * out_height * out_width;
            let workgroup_size = 256; // Match local_size_x from shader
            let num_workgroups: usize =
                (total_output_elements + workgroup_size - 1) / workgroup_size;

            self.device
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_matmul_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1_tensor: &ComputeTensor,
        src2_tensor: &ComputeTensor,
        dst_tensor: &ComputeTensor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src1_dims = src1_tensor.desc.to_dims();
        let src2_dims = src2_tensor.desc.to_dims();

        let operation = self.determine_matmul_variant(&src1_dims, &src2_dims);

        if operation == GPUMemoryOperation::MatMul {
            // Use generic fallback for unsupported dimension combinations
            self.create_generic_matmul_command_buffer(
                command_buffer,
                src1_tensor,
                src2_tensor,
                dst_tensor,
            )
        } else {
            // Use specialised implementation for supported dimensions
            self.create_specialized_matmul_command_buffer(
                command_buffer,
                src1_tensor,
                src2_tensor,
                dst_tensor,
                operation,
            )
        }
    }

    // Unified implementation for all specialised matmul operations
    fn create_specialized_matmul_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1_tensor: &ComputeTensor,
        src2_tensor: &ComputeTensor,
        dst_tensor: &ComputeTensor,
        operation: GPUMemoryOperation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let src1_mem = match &src1_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Source tensor 1 not in GPU memory".into()),
            };

            let src2_mem = match &src2_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Source tensor 2 not in GPU memory".into()),
            };

            let dst_mem = match &dst_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Destination tensor not in GPU memory".into()),
            };

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: src1_mem.buffer,
                    offset: 0,
                    range: src1_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: src2_mem.buffer,
                    offset: 0,
                    range: src2_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[2],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(operation)
                .ok_or(format!("{:?} pipeline not found", operation))?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            // Configure operation-specific parameters and dispatch dimensions
            let (push_constants, dispatch_x, dispatch_y, dispatch_z) =
                self.configure_matmul_operation(operation, src1_tensor, src2_tensor, dst_tensor)?;

            // Push constants to the shader
            self.device.cmd_push_constants(
                command_buffer,
                self.compute_pipelines.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    push_constants.as_ptr() as *const u8,
                    push_constants.len() * std::mem::size_of::<u32>(),
                ),
            );

            self.device
                .cmd_dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    fn configure_matmul_operation(
        &self,
        operation: GPUMemoryOperation,
        src1_tensor: &ComputeTensor,
        src2_tensor: &ComputeTensor,
        dst_tensor: &ComputeTensor,
    ) -> Result<(Vec<u32>, u32, u32, u32), Box<dyn std::error::Error>> {
        let src1_dims = src1_tensor.desc.to_dims();
        let src2_dims = src2_tensor.desc.to_dims();
        let dst_dims = dst_tensor.desc.to_dims();

        let src1_strides = src1_tensor.desc.strides();
        let src2_strides = src2_tensor.desc.strides();
        let dst_strides = dst_tensor.desc.strides();

        match operation {
            GPUMemoryOperation::MatMul1D2D => {
                // [k] × [k,n] → [n]
                let k = src1_dims[0];
                let n = src2_dims[1];

                let push_constants = vec![
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a
                    src2_strides[0] as u32, // stride_b0 (row stride)
                    src2_strides[1] as u32, // stride_b1 (column stride)
                    dst_strides[0] as u32,  // stride_c
                ];

                let workgroup_size = 256;
                let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;

                Ok((push_constants, num_groups_x, 1, 1))
            }

            GPUMemoryOperation::MatMul2D1D => {
                // [m,k] × [k] → [m]
                let m = src1_dims[0];
                let k = src1_dims[1];

                let push_constants = vec![
                    m as u32,               // m
                    k as u32,               // k
                    src1_strides[0] as u32, // stride_a0 (row stride)
                    src1_strides[1] as u32, // stride_a1 (column stride)
                    src2_strides[0] as u32, // stride_b
                    dst_strides[0] as u32,  // stride_c
                ];

                let workgroup_size = 256;
                let num_groups_x = (m as u32 + workgroup_size - 1) / workgroup_size;

                Ok((push_constants, num_groups_x, 1, 1))
            }

            GPUMemoryOperation::MatMul2D2D => {
                // [m,k] × [k,n] → [m,n]
                let m = src1_dims[0];
                let k = src1_dims[1];
                let n = src2_dims[1];

                let push_constants = vec![
                    m as u32,               // m
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a0 (row stride)
                    src1_strides[1] as u32, // stride_a1 (column stride)
                    src2_strides[0] as u32, // stride_b0 (row stride)
                    src2_strides[1] as u32, // stride_b1 (column stride)
                    dst_strides[0] as u32,  // stride_c0 (row stride)
                    dst_strides[1] as u32,  // stride_c1 (column stride)
                ];

                // Calculate workgroup dimensions - 16×16 threads per workgroup
                let workgroup_size = 16;
                let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;
                let num_groups_y = (m as u32 + workgroup_size - 1) / workgroup_size;

                Ok((push_constants, num_groups_x, num_groups_y, 1))
            }

            GPUMemoryOperation::MatMul2D3D => {
                // [m,k] × [batch,k,n] → [batch,m,n]
                let m = src1_dims[0];
                let k = src1_dims[1];
                let batch = src2_dims[0];
                let n = src2_dims[2];

                let push_constants = vec![
                    batch as u32,           // batch
                    m as u32,               // m
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a0 (row stride)
                    src1_strides[1] as u32, // stride_a1 (column stride)
                    src2_strides[0] as u32, // stride_b0 (batch stride)
                    src2_strides[1] as u32, // stride_b1 (row stride)
                    src2_strides[2] as u32, // stride_b2 (column stride)
                    dst_strides[0] as u32,  // stride_c0 (batch stride)
                    dst_strides[1] as u32,  // stride_c1 (row stride)
                    dst_strides[2] as u32,  // stride_c2 (column stride)
                ];

                // Calculate workgroup dimensions - 8×8×4 threads per workgroup
                let workgroup_size_xy = 8;
                let workgroup_size_z = 4;
                let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

                Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
            }

            GPUMemoryOperation::MatMul3D2D => {
                // [batch,m,k] × [k,n] → [batch,m,n]
                let batch = src1_dims[0];
                let m = src1_dims[1];
                let k = src1_dims[2];
                let n = src2_dims[1];

                let push_constants = vec![
                    batch as u32,           // batch
                    m as u32,               // m
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a0 (batch stride)
                    src1_strides[1] as u32, // stride_a1 (row stride)
                    src1_strides[2] as u32, // stride_a2 (column stride)
                    src2_strides[0] as u32, // stride_b0 (row stride)
                    src2_strides[1] as u32, // stride_b1 (column stride)
                    dst_strides[0] as u32,  // stride_c0 (batch stride)
                    dst_strides[1] as u32,  // stride_c1 (row stride)
                    dst_strides[2] as u32,  // stride_c2 (column stride)
                ];

                // Calculate workgroup dimensions - 8×8×4 threads per workgroup
                let workgroup_size_xy = 8;
                let workgroup_size_z = 4;
                let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

                Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
            }

            GPUMemoryOperation::MatMul3D3D => {
                // [batch,m,k] × [batch,k,n] → [batch,m,n]
                let batch = src1_dims[0];
                let m = src1_dims[1];
                let k = src1_dims[2];
                let n = src2_dims[2];

                let push_constants = vec![
                    batch as u32,           // batch
                    m as u32,               // m
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a0 (batch stride)
                    src1_strides[1] as u32, // stride_a1 (row stride)
                    src1_strides[2] as u32, // stride_a2 (column stride)
                    src2_strides[0] as u32, // stride_b0 (batch stride)
                    src2_strides[1] as u32, // stride_b1 (row stride)
                    src2_strides[2] as u32, // stride_b2 (column stride)
                    dst_strides[0] as u32,  // stride_c0 (batch stride)
                    dst_strides[1] as u32,  // stride_c1 (row stride)
                    dst_strides[2] as u32,  // stride_c2 (column stride)
                ];

                // Calculate workgroup dimensions - 8×8×4 threads per workgroup
                let workgroup_size_xy = 8;
                let workgroup_size_z = 4;
                let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
                let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

                Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
            }

            GPUMemoryOperation::MatMul3D1D => {
                // [batch,m,k] × [k] → [batch,m]
                let batch = src1_dims[0];
                let m = src1_dims[1];
                let k = src1_dims[2];

                let push_constants = vec![
                    batch as u32,           // batch
                    m as u32,               // m
                    k as u32,               // k
                    src1_strides[0] as u32, // stride_a0 (batch stride)
                    src1_strides[1] as u32, // stride_a1 (row stride)
                    src1_strides[2] as u32, // stride_a2 (column stride)
                    src2_strides[0] as u32, // stride_b
                    dst_strides[0] as u32,  // stride_c0 (batch stride)
                    dst_strides[1] as u32,  // stride_c1 (row stride)
                ];

                // Calculate workgroup dimensions - 16×16 threads per workgroup
                let workgroup_size = 16;
                let num_groups_x = (m as u32 + workgroup_size - 1) / workgroup_size;
                let num_groups_y = (batch as u32 + workgroup_size - 1) / workgroup_size;

                Ok((push_constants, num_groups_x, num_groups_y, 1))
            }

            GPUMemoryOperation::MatMul1D3D => {
                // [k] × [batch,k,n] → [batch,n]
                let k = src1_dims[0];
                let batch = src2_dims[0];
                let n = src2_dims[2];

                let push_constants = vec![
                    batch as u32,           // batch
                    k as u32,               // k
                    n as u32,               // n
                    src1_strides[0] as u32, // stride_a
                    src2_strides[0] as u32, // stride_b0 (batch stride)
                    src2_strides[1] as u32, // stride_b1 (row stride)
                    src2_strides[2] as u32, // stride_b2 (column stride)
                    dst_strides[0] as u32,  // stride_c0 (batch stride)
                    dst_strides[1] as u32,  // stride_c1 (column stride)
                ];

                // Calculate workgroup dimensions - 16×16 threads per workgroup
                let workgroup_size = 16;
                let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;
                let num_groups_y = (batch as u32 + workgroup_size - 1) / workgroup_size;

                Ok((push_constants, num_groups_x, num_groups_y, 1))
            }

            _ => Err(format!(
                "Unsupported operation in configure_matmul_operation: {:?}",
                operation
            )
            .into()),
        }
    }

    pub fn create_generic_matmul_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src1_tensor: &ComputeTensor,
        src2_tensor: &ComputeTensor,
        dst_tensor: &ComputeTensor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let src1_mem = match &src1_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Source tensor 1 not in GPU memory".into()),
            };

            let src2_mem = match &src2_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Source tensor 2 not in GPU memory".into()),
            };

            let dst_mem = match &dst_tensor.data {
                TensorData::GPU { memory, .. } => memory,
                _ => return Err("Destination tensor not in GPU memory".into()),
            };

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: src1_mem.buffer,
                    offset: 0,
                    range: src1_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: src2_mem.buffer,
                    offset: 0,
                    range: src2_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[2],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = self
                .compute_pipelines
                .get_pipeline(GPUMemoryOperation::MatMul)
                .ok_or("Generic MatMul pipeline not found")?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let src1_dims = src1_tensor.desc.to_dims();
            let src2_dims = src2_tensor.desc.to_dims();
            let dst_dims = dst_tensor.desc.to_dims();

            let src1_strides = src1_tensor.desc.strides();
            let src2_strides = src2_tensor.desc.strides();
            let dst_strides = dst_tensor.desc.strides();

            const MAX_DIMS: usize = 8;

            // Analyse tensors to find matrix multiplication dimensions
            let (m, k, n, a_m_axis, a_k_axis, b_k_axis, b_n_axis) =
                self.analyze_matmul_dimensions(&src1_dims, &src2_dims, &dst_dims)?;

            // Prepare push constants with tensor information
            let mut push_constants = Vec::with_capacity(3 + 3 + MAX_DIMS * 6);

            // Dimension counts
            push_constants.push(src1_dims.len() as u32);
            push_constants.push(src2_dims.len() as u32);
            push_constants.push(dst_dims.len() as u32);

            // Key dimensions
            push_constants.push(m as u32);
            push_constants.push(k as u32);
            push_constants.push(n as u32);

            // Tensor A shape and strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < src1_dims.len() {
                    src1_dims[i] as u32
                } else {
                    1
                });
            }

            // Tensor B shape and strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < src2_dims.len() {
                    src2_dims[i] as u32
                } else {
                    1
                });
            }

            // Tensor C shape and strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < dst_dims.len() {
                    dst_dims[i] as u32
                } else {
                    1
                });
            }

            // Tensor A strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < src1_strides.len() {
                    src1_strides[i] as u32
                } else {
                    0
                });
            }

            // Tensor B strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < src2_strides.len() {
                    src2_strides[i] as u32
                } else {
                    0
                });
            }

            // Tensor C strides
            for i in 0..MAX_DIMS {
                push_constants.push(if i < dst_strides.len() {
                    dst_strides[i] as u32
                } else {
                    0
                });
            }

            // Axis information
            push_constants.push(a_m_axis as u32);
            push_constants.push(a_k_axis as u32);
            push_constants.push(b_k_axis as u32);
            push_constants.push(b_n_axis as u32);

            // Push constants to the shader
            self.device.cmd_push_constants(
                command_buffer,
                self.compute_pipelines.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    push_constants.as_ptr() as *const u8,
                    push_constants.len() * std::mem::size_of::<u32>(),
                ),
            );

            // Calculate batch size (product of all dimensions except m and n)
            let mut batch_size = 1;
            for (i, &dim) in dst_dims.iter().enumerate() {
                if i != a_m_axis && i != b_n_axis {
                    batch_size *= dim;
                }
            }

            // Calculate dispatch size
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let workgroup_size_z = 4;

            let num_groups_x = (n as u32 + workgroup_size_x - 1) / workgroup_size_x;
            let num_groups_y = (m as u32 + workgroup_size_y - 1) / workgroup_size_y;
            let num_groups_z = (batch_size as u32 + workgroup_size_z - 1) / workgroup_size_z;

            self.device
                .cmd_dispatch(command_buffer, num_groups_x, num_groups_y, num_groups_z);

            self.device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    fn analyze_matmul_dimensions(
        &self,
        src1_dims: &[usize],
        src2_dims: &[usize],
        dst_dims: &[usize],
    ) -> Result<(usize, usize, usize, usize, usize, usize, usize), Box<dyn std::error::Error>> {
        if src1_dims.is_empty() || src2_dims.is_empty() {
            return Err("Empty tensor dimensions".into());
        }

        // Find matrix multiplication dimensions based on common patterns

        // Pattern 1: Standard matrix multiplication [m,k] × [k,n] → [m,n]
        if src1_dims.len() == 2 && src2_dims.len() == 2 && dst_dims.len() == 2 {
            let m = src1_dims[0];
            let k1 = src1_dims[1];
            let k2 = src2_dims[0];
            let n = src2_dims[1];

            if k1 == k2 && dst_dims[0] == m && dst_dims[1] == n {
                return Ok((m, k1, n, 0, 1, 0, 1));
            }
        }

        // Pattern 2: Batched matrix multiplication [batch,m,k] × [batch,k,n] → [batch,m,n]
        if src1_dims.len() == 3 && src2_dims.len() == 3 && dst_dims.len() == 3 {
            let batch1 = src1_dims[0];
            let m = src1_dims[1];
            let k1 = src1_dims[2];

            let batch2 = src2_dims[0];
            let k2 = src2_dims[1];
            let n = src2_dims[2];

            if batch1 == batch2
                && k1 == k2
                && dst_dims[0] == batch1
                && dst_dims[1] == m
                && dst_dims[2] == n
            {
                return Ok((m, k1, n, 1, 2, 1, 2));
            }
        }

        // Pattern 3: Higher-dimensional tensor contraction - general case
        // This is a complex analysis that would try to find matching dimensions
        // For now, let's implement a simplified version for common cases

        // Find the innermost dimensions, which are likely the matrix multiply dimensions
        let a_k_axis = src1_dims.len() - 1;
        let b_k_axis = src2_dims.len() - 2;
        let a_m_axis = src1_dims.len() - 2;
        let b_n_axis = src2_dims.len() - 1;

        let k1 = src1_dims[a_k_axis];
        let k2 = src2_dims[b_k_axis];

        if k1 != k2 {
            return Err(format!(
                "Inner dimensions for matrix multiplication don't match: {} vs {}",
                k1, k2
            )
            .into());
        }

        let m = src1_dims[a_m_axis];
        let n = src2_dims[b_n_axis];

        // Check that output shape is compatible
        if dst_dims.len() < 2
            || dst_dims[dst_dims.len() - 2] != m
            || dst_dims[dst_dims.len() - 1] != n
        {
            return Err(format!(
                "Output shape {:?} doesn't match expected dimensions m={}, n={}",
                dst_dims, m, n
            )
            .into());
        }

        // For now, we'll assume a standard matmul pattern, but we can expand this
        // to handle more complex contractions if needed
        Ok((m, k1, n, a_m_axis, a_k_axis, b_k_axis, b_n_axis))
    }

    pub fn determine_matmul_variant(
        &self,
        src1_dims: &[usize],
        src2_dims: &[usize],
    ) -> GPUMemoryOperation {
        match (src1_dims.len(), src2_dims.len()) {
            (1, 2) => GPUMemoryOperation::MatMul1D2D,
            (2, 1) => GPUMemoryOperation::MatMul2D1D,
            (2, 2) => GPUMemoryOperation::MatMul2D2D,
            (2, 3) => GPUMemoryOperation::MatMul2D3D,
            (3, 2) => GPUMemoryOperation::MatMul3D2D,
            (3, 3) => GPUMemoryOperation::MatMul3D3D,
            (3, 1) => GPUMemoryOperation::MatMul3D1D,
            (1, 3) => GPUMemoryOperation::MatMul1D3D,
            _ => {
                // Fallback to generic, or error
                eprintln!(
                    "Unsupported tensor dimensions for matmul: {:?} × {:?}",
                    src1_dims, src2_dims
                );
                GPUMemoryOperation::MatMul // Fall back to generic (which will likely fail)
            }
        }
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
