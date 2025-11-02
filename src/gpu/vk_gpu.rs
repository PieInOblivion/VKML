use std::{
    collections::{HashMap, HashSet},
    ffi::{CString, c_void},
    ptr,
    sync::{Arc, Mutex, OnceLock, RwLock},
};
use vulkanalia::{
    Device, Instance,
    vk::{
        self, DeviceV1_0, DeviceV1_3, Handle, InstanceV1_0, InstanceV1_1,
        KhrPushDescriptorExtensionDeviceCommands,
    },
};

use crate::{
    compute::memory_tracker::MemoryTracker,
    gpu::workgroup::optimal_workgroup_size,
    instruction::GPUOperation,
    utils::{error::VKMLError, expect_msg::ExpectMsg},
};

use super::gpu_memory::GPUMemory;
use super::vk_extensions::VkExtensions;

pub struct Gpu {
    properties: vk::PhysicalDeviceProperties,
    subgroup_properties: vk::PhysicalDeviceSubgroupProperties,
    push_descriptor_properties: vk::PhysicalDevicePushDescriptorPropertiesKHR,

    has_compute: bool,
    max_compute_queue_count: u32,
    host_visible_device_local_bytes: u64, // bytes available on the device that satisfy DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT

    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    memory_tracker: MemoryTracker,
    extensions: VkExtensions,

    // staging buffer used for host -> device-only transfers
    staging_buffer: OnceLock<Mutex<GPUMemory>>,

    // Drop order matters: fields drop top-to-bottom
    pipelines: RwLock<HashMap<(GPUOperation, [u32; 3], usize), vk::Pipeline>>,
    descriptor_set_layouts: Box<[OnceLock<vk::DescriptorSetLayout>]>,
    pipeline_layouts: Box<[OnceLock<vk::PipelineLayout>]>,
    command_pool: vk::CommandPool,
    device: Arc<Device>,
    instance: Arc<Instance>,
}

impl Gpu {
    pub fn new_shared(
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, VKMLError> {
        unsafe {
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let device_extensions =
                instance.enumerate_device_extension_properties(physical_device, None)?;
            let vk_extensions = VkExtensions::from_extension_properties(
                &instance,
                physical_device,
                &device_extensions,
            )?;

            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .expect("No compute queue family found on device");

            // Request a single compute queue
            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count: 1,
                queue_priorities: &1.0f32,
            };

            let device_features = vk::PhysicalDeviceFeatures::default();

            // vkextensions to prepare both extension name strings and any required
            // p_next feature structs. keep both return values alive until after device creation.
            let extras = vk_extensions.prepare_device_create();
            let enabled_names_ptrs = &extras.name_ptrs;
            let device_create_pnext = extras.device_create_next();

            // create device with requested queue count
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                next: device_create_pnext,
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: 1,
                queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                enabled_layer_names: std::ptr::null(),
                enabled_extension_count: enabled_names_ptrs.len() as u32,
                enabled_extension_names: if enabled_names_ptrs.is_empty() {
                    ptr::null()
                } else {
                    enabled_names_ptrs.as_ptr()
                },
                enabled_features: &device_features,
            };
            // Note: keep 'extras' alive until after device creation so its CStrings and
            // p_next owners remain valid for the call.
            let device =
                Arc::new(instance.create_device(physical_device, &device_create_info, None)?);

            // Query device properties
            let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
            let mut push_props = vk::PhysicalDevicePushDescriptorPropertiesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
                next: &mut subgroup_properties as *mut _ as *mut c_void,
                max_push_descriptors: 0,
            };

            let mut props2 = vk::PhysicalDeviceProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
                next: &mut push_props as *mut _ as *mut c_void,
                properties: Default::default(),
            };

            instance.get_physical_device_properties2(physical_device, &mut props2);
            let properties = props2.properties;

            // Memory properties, memory_budget if extension is available
            let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT {
                s_type: vk::StructureType::PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
                next: std::ptr::null_mut(),
                heap_budget: [0; vk::MAX_MEMORY_HEAPS],
                heap_usage: [0; vk::MAX_MEMORY_HEAPS],
            };

            let mut memory_props2 = vk::PhysicalDeviceMemoryProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                next: if vk_extensions.has_memory_budget() {
                    &mut budget_props as *mut _ as *mut c_void
                } else {
                    std::ptr::null_mut()
                },
                memory_properties: Default::default(),
            };

            instance.get_physical_device_memory_properties2(physical_device, &mut memory_props2);
            let memory_properties = memory_props2.memory_properties;

            // Compute how many bytes are available in memory types that are
            // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT.
            let mut hv_dl_heap_indices = HashSet::new();
            for i in 0..memory_properties.memory_type_count {
                let mem_type = memory_properties.memory_types[i as usize];
                if mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && mem_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && mem_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                {
                    hv_dl_heap_indices.insert(mem_type.heap_index as usize);
                }
            }

            let mut hv_dl_bytes: u128 = 0;
            for &heap_idx in hv_dl_heap_indices.iter() {
                if vk_extensions.has_memory_budget() {
                    hv_dl_bytes += budget_props.heap_budget[heap_idx] as u128;
                } else {
                    hv_dl_bytes += memory_properties.memory_heaps[heap_idx].size as u128;
                }
            }

            let host_visible_device_local_bytes = hv_dl_bytes as u64;

            // Get the single compute queue
            let compute_queue = device.get_device_queue(queue_family_index, 0);

            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                next: ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
            };

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            // caches for descriptor set layouts and pipeline layouts indexed by binding count
            // uses max_push_descriptors as a reasonable upper bound
            let max_bindings = push_props.max_push_descriptors as usize;
            let descriptor_set_layouts = (0..=max_bindings)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let pipeline_layouts = (0..=max_bindings)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice();

            // Check compute capability
            let (has_compute, max_compute_queue_count) = queue_families
                .iter()
                .find(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|props| (true, props.queue_count))
                .unwrap_or((false, 0));

            // Calculate available memory budget
            let device_local_heap_index = (0..memory_properties.memory_type_count)
                .find(|&i| {
                    let memory_type = memory_properties.memory_types[i as usize];
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .map(|i| memory_properties.memory_types[i as usize].heap_index)
                .unwrap_or(0);

            let memory_budget = if vk_extensions.has_memory_budget() {
                // use actual budget from VK_EXT_memory_budget, scaled to 95% to account for overhead
                // use u128 to avoid overflow when multiplying large u64 values
                let reported_budget =
                    budget_props.heap_budget[device_local_heap_index as usize] as u128;
                ((reported_budget * 95) / 100) as u64
            } else {
                // use 80% of total device memory
                let total_memory =
                    memory_properties.memory_heaps[device_local_heap_index as usize].size as u128;
                ((total_memory * 80) / 100) as u64
            };

            Ok(Self {
                properties,
                subgroup_properties,
                push_descriptor_properties: push_props,

                has_compute,
                max_compute_queue_count,
                host_visible_device_local_bytes,

                physical_device,
                compute_queue,
                memory_tracker: MemoryTracker::new(memory_budget),
                extensions: vk_extensions,

                staging_buffer: OnceLock::new(),

                pipelines: RwLock::new(HashMap::new()),
                descriptor_set_layouts,
                pipeline_layouts,
                command_pool,
                device,
                instance,
            })
        }
    }

    /// Move some raw bytes into a host-visible/coherent device allocation and return it.
    /// The returned GPUMemory will be mappable by the CPU.
    pub fn move_to_gpu_host_visible(&self, bytes: &[u8]) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes.len() as vk::DeviceSize;
        self.memory_tracker.allocate(size_in_bytes);

        // host visible path create mappable buffer and copy directly
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
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            // Map memory and write raw bytes
            let data_ptr =
                self.device
                    .map_memory(memory, 0, size_in_bytes, vk::MemoryMapFlags::empty())?
                    as *mut u8;

            // Copy the data
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());

            self.device.unmap_memory(memory);

            Ok(GPUMemory::new(
                buffer,
                memory,
                size_in_bytes,
                self.device.clone(),
            ))
        }
    }

    /// Move raw bytes into a device-local-only allocation using the internal staging buffer
    pub fn move_to_gpu_host_not_visible(&self, bytes: &[u8]) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes.len() as vk::DeviceSize;
        self.memory_tracker.allocate(size_in_bytes);

        // Device-local path: allocate a DEVICE_LOCAL buffer and copy into it using the staging buffer
        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: ptr::null(),
            };

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);

            let memory_type = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            let dest = GPUMemory::new(buffer, memory, size_in_bytes, self.device.clone());

            // Copy in chunks via staging buffer. Hold the staging mutex for the full transfer
            // so other threads can't overwrite the staging buffer between chunk writes.
            let staging_mutex = self.get_or_create_staging_buffer();
            let staging_guard = staging_mutex.lock().unwrap();
            let staging_size = staging_guard.size as usize;

            let mut offset_usize = 0usize;

            // single reusable fence for per-chunk waits
            let fence_info = vk::FenceCreateInfo {
                s_type: vk::StructureType::FENCE_CREATE_INFO,
                next: ptr::null(),
                flags: vk::FenceCreateFlags::empty(),
            };

            let fence = self.device.create_fence(&fence_info, None)?;

            while offset_usize < bytes.len() {
                let remaining = bytes.len() - offset_usize;
                let chunk_size = std::cmp::min(staging_size, remaining);
                let chunk = &bytes[offset_usize..offset_usize + chunk_size];

                // copy into staging (maps and writes)
                staging_guard.copy_into(chunk).map_err(|e| {
                    VKMLError::Vulkan(format!("Failed to copy into staging buffer: {}", e))
                })?;

                // allocate a temporary command buffer and record a copy from staging -> dest
                let alloc_info = vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    next: ptr::null(),
                    command_pool: self.get_command_pool(),
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                };

                let buffers = self.device.allocate_command_buffers(&alloc_info)?;
                let command_buffer = buffers.into_iter().next().ok_or_else(|| {
                    VKMLError::Vulkan("No command buffer returned for staging copy".to_string())
                })?;

                self.begin_command_buffer(command_buffer)?;

                let copy_region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset_usize as vk::DeviceSize,
                    size: chunk_size as vk::DeviceSize,
                };

                self.get_device().cmd_copy_buffer(
                    command_buffer,
                    staging_guard.buffer,
                    dest.buffer,
                    &[copy_region],
                );

                self.end_command_buffer(command_buffer)?;

                // Submit and wait for completion
                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    next: ptr::null(),
                    wait_semaphore_count: 0,
                    wait_semaphores: ptr::null(),
                    wait_dst_stage_mask: ptr::null(),
                    command_buffer_count: 1,
                    command_buffers: &command_buffer,
                    signal_semaphore_count: 0,
                    signal_semaphores: ptr::null(),
                };

                // submit using the reusable fence
                self.device
                    .queue_submit(self.compute_queue, &[submit_info], fence)?;

                // wait for this fence to be signalled, copy finished
                self.device.wait_for_fences(&[fence], true, u64::MAX)?;

                // reusable fence to unsignalled state
                self.device.reset_fences(&[fence])?;

                offset_usize += chunk_size;
            }

            Ok(dest)
        }
    }

    /// Allocate uninitialised GPU memory. If requires_host_visability is true the allocation
    /// will be host-visible/host-coherent (mappable). Otherwise it will be DEVICE_LOCAL only.
    pub fn allocate_uninitialised_gpu_memory(
        &self,
        bytes: usize,
        requires_host_visability: bool,
    ) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes as vk::DeviceSize;
        self.memory_tracker.allocate(size_in_bytes);

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

            let properties = if requires_host_visability {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };

            let memory_type = self.find_memory_type(mem_requirements.memory_type_bits, properties);

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;

            self.device.bind_buffer_memory(buffer, memory, 0)?;

            Ok(GPUMemory::new(
                buffer,
                memory,
                size_in_bytes,
                self.device.clone(),
            ))
        }
    }

    /// Lazily create (and return) the staging buffer mutex. Staging buffer is host-visible and sized
    /// to 5% of the tracked maximum memory.
    pub fn get_or_create_staging_buffer(&self) -> &Mutex<GPUMemory> {
        self.staging_buffer.get_or_init(|| unsafe {
            // size: 5% of total memory
            let staging_size = (self.memory_total() / 20) as usize;

            // Account for the staging buffer in the memory tracker
            self.memory_tracker.allocate(staging_size as vk::DeviceSize);

            // Create a host-visible buffer usable as transfer source.
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: staging_size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: ptr::null(),
            };

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .expect_msg("Failed to create staging buffer");
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);

            // Classic staging buffer design states that we give the memory to vulkan on the cpu side,
            // then use that to transfer into the gpu memory.
            // However if the device has enough memory that satisfies host visable, coherent and device local,
            // such as rebar gpus, then we can put the staging buffer into the gpu itself.
            // Idealy we wouldn't use stages at all if there is enough.
            // Might resimplify and only support rebar devices in future.

            // Might also limit it to a min of the supported 3 flags memory total or 5% total, likely requires
            // a second magic number minimum or something though. This would allow us to use the typical 255mb limit of
            // non-rebar gpus, but need to account for gpus where 255mb > 5% total, or where the limit could be 5mb and
            // we should use cpu.
            let requested_properties =
                if self.host_visible_device_local_bytes >= staging_size as u64 {
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::DEVICE_LOCAL
                } else {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                };

            let memory_type =
                self.find_memory_type(mem_requirements.memory_type_bits, requested_properties);

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self
                .device
                .allocate_memory(&alloc_info, None)
                .expect_msg("Failed to allocate staging memory");
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .expect_msg("Failed to bind staging buffer memory");

            let staging = GPUMemory::new(
                buffer,
                memory,
                staging_size as vk::DeviceSize,
                self.device.clone(),
            );

            Mutex::new(staging)
        })
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

    fn create_pipeline(
        &self,
        shader_code: &[u8],
        local_size: [u32; 3],
        binding_count: usize,
    ) -> Result<vk::Pipeline, VKMLError> {
        unsafe {
            // ensure the shader byte length is a multiple of 4 (SPIR-V is in 32-bit words)
            if !shader_code.len().is_multiple_of(4) {
                return Err(VKMLError::Vulkan(
                    "shader byte length must be a multiple of 4".to_string(),
                ));
            }

            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::ShaderModuleCreateFlags::empty(),
                code_size: shader_code.len(),
                code: shader_code.as_ptr() as *const u32,
            };

            let shader_module = self.device.create_shader_module(&shader_info, None)?;

            let entry_point = CString::new("main").unwrap();

            // Prepare specialization constants so shaders compiled to use
            // specialization IDs 0,1,2 for local_size_x/y/z will receive
            // the chosen local workgroup sizes at pipeline creation time.
            let spec_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4,
                },
            ];

            let spec_info = vk::SpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                map_entries: spec_entries.as_ptr(),
                data_size: (local_size.len() * std::mem::size_of::<u32>()),
                data: local_size.as_ptr() as *const c_void,
            };

            let pipeline_layout = self.get_pipeline_layout(binding_count);

            let pipeline_info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::PipelineCreateFlags::empty(),
                stage: vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    next: std::ptr::null(),
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module: shader_module,
                    name: entry_point.as_ptr(),
                    specialization_info: &spec_info,
                },
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: -1,
            };

            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
                .0[0];

            self.device.destroy_shader_module(shader_module, None);

            Ok(pipeline)
        }
    }

    pub fn get_or_create_pipeline(
        &self,
        op: GPUOperation,
        local_size: [u32; 3],
        binding_count: usize,
    ) -> vk::Pipeline {
        let key = (op, local_size, binding_count);

        // Fast path: pipeline already exists
        if let Some(pipeline) = self.pipelines.read().unwrap().get(&key) {
            return *pipeline;
        }

        // Slow path: create pipeline
        let pipeline = self
            .create_pipeline(op.get_shader_bytes(), local_size, binding_count)
            .expect_msg(&format!(
                "Pipeline creation failed {:?} with workgroup {:?}",
                op, local_size
            ));

        // Insert and return (handles race condition gracefully)
        self.pipelines
            .write()
            .unwrap()
            .entry(key)
            .or_insert(pipeline);
        pipeline
    }

    pub fn get_pipeline_layout(&self, binding_count: usize) -> vk::PipelineLayout {
        assert!(
            binding_count < self.pipeline_layouts.len(),
            "Binding count {} exceeds maximum {}",
            binding_count,
            self.pipeline_layouts.len() - 1
        );

        // Get or create the pipeline layout which internally gets/creates descriptor set layout
        *self.pipeline_layouts[binding_count].get_or_init(|| unsafe {
            let descriptor_set_layout = self.get_descriptor_set_layout(binding_count);

            // 128 bytes is the minimum guaranteed push constant space for the vulkan spec
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 128,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: 1,
                set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                push_constant_ranges: &push_constant_range,
            };

            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        })
    }

    fn get_descriptor_set_layout(&self, binding_count: usize) -> vk::DescriptorSetLayout {
        *self.descriptor_set_layouts[binding_count].get_or_init(|| unsafe {
            // N identical storage buffer bindings
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
                .map(|i| vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                })
                .collect();

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR,
                binding_count: bindings.len() as u32,
                bindings: bindings.as_ptr(),
            };

            self.device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .expect("Failed to create descriptor set layout")
        })
    }

    pub fn extensions(&self) -> &VkExtensions {
        &self.extensions
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_properties.subgroup_size
    }

    pub fn subgroup_supported_operations(&self) -> vk::SubgroupFeatureFlags {
        self.subgroup_properties.supported_operations
    }

    pub fn host_visible_device_local_bytes(&self) -> u64 {
        self.host_visible_device_local_bytes
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn memory_total(&self) -> u64 {
        self.memory_tracker.get_maximum()
    }

    pub fn memory_available(&self) -> u64 {
        self.memory_tracker.get_available()
    }

    pub fn memory_usage(&self) -> u64 {
        self.memory_tracker.get_current()
    }

    pub fn name(&self) -> String {
        String::from_utf8_lossy(
            &self
                .properties
                .device_name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect::<Vec<u8>>(),
        )
        .to_string()
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.device_type
    }

    pub fn has_compute(&self) -> bool {
        self.has_compute
    }

    pub fn max_workgroup_count(&self) -> [u32; 3] {
        self.properties.limits.max_compute_work_group_count
    }

    pub fn max_workgroup_size(&self) -> [u32; 3] {
        self.properties.limits.max_compute_work_group_size
    }

    pub fn max_workgroup_invocations(&self) -> u32 {
        self.properties.limits.max_compute_work_group_invocations
    }

    pub fn max_compute_queue_count(&self) -> u32 {
        self.max_compute_queue_count
    }

    pub fn max_shared_memory_size(&self) -> u32 {
        self.properties.limits.max_compute_shared_memory_size
    }

    pub fn max_push_descriptors(&self) -> u32 {
        self.push_descriptor_properties.max_push_descriptors
    }

    pub fn get_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn get_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Calculate optimal workgroup size for 1D compute operations (element-wise ops)
    pub fn optimal_workgroup_size_1d(&self, total_elements: u64) -> [u32; 3] {
        optimal_workgroup_size(
            self.max_workgroup_size(),
            self.max_workgroup_invocations(),
            [Some(total_elements), None, None],
        )
    }

    /// Calculate optimal workgroup size for 2D compute operations (matmul, conv2d)
    ///
    /// Note: For batched 2D operations (e.g., batched matmul), use this function
    /// and handle the batch dimension in the dispatch call, not the workgroup size.
    /// Standard pattern: workgroup = [tile, tile], dispatch = [m/tile, n/tile, batch]
    pub fn optimal_workgroup_size_2d(&self, rows: u64, cols: u64) -> [u32; 3] {
        optimal_workgroup_size(
            self.max_workgroup_size(),
            self.max_workgroup_invocations(),
            [Some(rows), Some(cols), None],
        )
    }

    /// Calculate optimal workgroup size for 3D spatial operations (conv3d, maxpool3d)
    pub fn optimal_workgroup_size_3d(&self, x: u64, y: u64, z: u64) -> [u32; 3] {
        optimal_workgroup_size(
            self.max_workgroup_size(),
            self.max_workgroup_invocations(),
            [Some(x), Some(y), Some(z)],
        )
    }

    /// Begin recording a compute command buffer
    pub fn begin_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<(), VKMLError> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            self.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;
        }
        Ok(())
    }

    /// Bind GPU storage buffers to descriptor set bindings
    pub fn bind_storage_buffers(&self, command_buffer: vk::CommandBuffer, buffers: &[&GPUMemory]) {
        unsafe {
            let buffer_infos: Vec<_> = buffers
                .iter()
                .map(|mem| vk::DescriptorBufferInfo {
                    buffer: mem.buffer,
                    offset: 0,
                    range: mem.size,
                })
                .collect();

            let write_descriptor_sets: Vec<_> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: info,
                    image_info: ptr::null(),
                    texel_buffer_view: ptr::null(),
                })
                .collect();

            self.get_device().cmd_push_descriptor_set_khr(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.get_pipeline_layout(buffers.len()),
                0,
                &write_descriptor_sets,
            );
        }
    }

    /// Insert a memory barrier ensuring previous compute writes are visible to subsequent compute dispatches.
    pub fn barrier_compute_shader_access(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer_barriers: &[vk::BufferMemoryBarrier2],
    ) {
        unsafe {
            let dependency_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                next: ptr::null(),
                dependency_flags: vk::DependencyFlags::empty(),
                memory_barrier_count: 0,
                memory_barriers: ptr::null(),
                buffer_memory_barrier_count: buffer_barriers.len() as u32,
                buffer_memory_barriers: buffer_barriers.as_ptr(),
                image_memory_barrier_count: 0,
                image_memory_barriers: ptr::null(),
            };

            self.get_device()
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }
    }

    /// Bind GPU storage buffers (supporting Option<&GPUMemory>) to descriptor set bindings
    /// Optional buffers will be bound as null buffers with size 0
    pub fn bind_storage_buffers_optional(
        &self,
        command_buffer: vk::CommandBuffer,
        buffers: &[Option<&GPUMemory>],
    ) {
        unsafe {
            let buffer_infos: Vec<_> = buffers
                .iter()
                .map(|mem_opt| {
                    if let Some(mem) = mem_opt {
                        vk::DescriptorBufferInfo {
                            buffer: mem.buffer,
                            offset: 0,
                            range: mem.size,
                        }
                    } else {
                        vk::DescriptorBufferInfo {
                            buffer: vk::Buffer::null(),
                            offset: 0,
                            range: 0,
                        }
                    }
                })
                .collect();

            let write_descriptor_sets: Vec<_> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: info,
                    image_info: ptr::null(),
                    texel_buffer_view: ptr::null(),
                })
                .collect();

            self.get_device().cmd_push_descriptor_set_khr(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.get_pipeline_layout(buffers.len()),
                0,
                &write_descriptor_sets,
            );
        }
    }

    /// Bind a compute pipeline with the specified workgroup size and binding count
    pub fn bind_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        operation: GPUOperation,
        local_size: [u32; 3],
        binding_count: usize,
    ) {
        unsafe {
            let pipeline = self.get_or_create_pipeline(operation, local_size, binding_count);
            self.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    /// Push constants to the compute shader
    pub fn bind_push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        binding_count: usize,
        data: &[u8],
    ) {
        unsafe {
            self.get_device().cmd_push_constants(
                command_buffer,
                self.get_pipeline_layout(binding_count),
                vk::ShaderStageFlags::COMPUTE,
                0,
                data,
            );
        }
    }

    /// Dispatch compute shader work
    pub fn dispatch(&self, cb: vk::CommandBuffer, local_size: [u32; 3], work_size: [u64; 3]) {
        let dispatch_x = work_size[0].div_ceil(local_size[0] as u64) as u32;
        let dispatch_y = work_size[1].div_ceil(local_size[1] as u64) as u32;
        let dispatch_z = work_size[2].div_ceil(local_size[2] as u64) as u32;
        unsafe {
            self.get_device()
                .cmd_dispatch(cb, dispatch_x, dispatch_y, dispatch_z);
        }
    }

    /// End command buffer recording
    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<(), VKMLError> {
        unsafe {
            self.get_device().end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    pub fn submit_with_fence(
        &self,
        command_buffers: &[vk::CommandBuffer],
        fence: Option<vk::Fence>,
    ) -> Result<(), VKMLError> {
        unsafe {
            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                next: ptr::null(),
                wait_semaphore_count: 0,
                wait_semaphores: ptr::null(),
                wait_dst_stage_mask: ptr::null(),
                command_buffer_count: command_buffers.len() as u32,
                command_buffers: command_buffers.as_ptr(),
                signal_semaphore_count: 0,
                signal_semaphores: ptr::null(),
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                fence.unwrap_or(vk::Fence::null()),
            )?;
        }

        Ok(())
    }

    pub fn wait_and_reset_fence(&self, fence: vk::Fence) -> Result<(), VKMLError> {
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
        }

        Ok(())
    }

    pub fn create_fence(&self) -> Result<vk::Fence, VKMLError> {
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            next: ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };

        unsafe { Ok(self.device.create_fence(&fence_info, None)?) }
    }
}
