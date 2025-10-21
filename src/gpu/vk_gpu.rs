use std::{
    array,
    collections::HashMap,
    ffi::{CString, c_void},
    ptr,
    sync::{
        Arc, OnceLock, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};
use vulkanalia::{
    Device, Instance,
    vk::{
        self, DeviceV1_0, DeviceV1_2, Handle, InstanceV1_0, InstanceV1_1,
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
    name: String,
    device_type: vk::PhysicalDeviceType,
    has_compute: bool,
    max_workgroup_count: [u32; 3],
    max_workgroup_size: [u32; 3],
    max_workgroup_invocations: u32,
    max_shared_memory_size: u32,
    max_compute_queue_count: u32,
    max_push_descriptors: u32,
    subgroup_size: u32, // eg: 32 for NVIDIA/Intel, 64 for AMD

    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    descriptor_set_layout: vk::DescriptorSetLayout,
    memory_tracker: MemoryTracker,
    extensions: VkExtensions,

    // Drop order matters: fields drop top-to-bottom
    timeline_semaphore: OnceLock<vk::Semaphore>,
    next_semaphore_value: AtomicU64,
    pipelines: RwLock<HashMap<(GPUOperation, [u32; 3]), vk::Pipeline>>,
    pipeline_layout: vk::PipelineLayout,
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
            let subgroup_size = subgroup_properties.subgroup_size;

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

            // Get the single compute queue
            let compute_queue = device.get_device_queue(queue_family_index, 0);

            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                next: ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
            };

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            // Create 4 identical storage buffer bindings
            let bindings: [vk::DescriptorSetLayoutBinding; 4] =
                array::from_fn(|i| vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                });

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR,
                binding_count: bindings.len() as u32,
                bindings: bindings.as_ptr(),
            };

            let descriptor_set_layout =
                device.create_descriptor_set_layout(&descriptor_layout_info, None)?;

            // 128 bytes is the minimum guaranteed push constant space for the vulkan spec
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 128,
            };

            let pipeline_layout = {
                let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                    s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                    next: std::ptr::null(),
                    flags: vk::PipelineLayoutCreateFlags::empty(),
                    set_layout_count: 1,
                    set_layouts: &descriptor_set_layout,
                    push_constant_range_count: 1,
                    push_constant_ranges: &push_constant_range,
                };

                device.create_pipeline_layout(&pipeline_layout_info, None)?
            };

            let name = String::from_utf8_lossy(
                &properties
                    .device_name
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect::<Vec<u8>>(),
            )
            .to_string();

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
                name,
                device_type: properties.device_type,
                has_compute,
                max_workgroup_count: properties.limits.max_compute_work_group_count,
                max_workgroup_size: properties.limits.max_compute_work_group_size,
                max_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
                max_shared_memory_size: properties.limits.max_compute_shared_memory_size,
                max_compute_queue_count,
                max_push_descriptors: push_props.max_push_descriptors,
                subgroup_size,

                physical_device,
                compute_queue,
                descriptor_set_layout,
                memory_tracker: MemoryTracker::new(memory_budget),
                extensions: vk_extensions,
                timeline_semaphore: OnceLock::new(),
                next_semaphore_value: AtomicU64::new(1),

                pipelines: RwLock::new(HashMap::new()),
                pipeline_layout,
                command_pool,
                device,
                instance,
            })
        }
    }

    pub fn move_to_gpu(&self, bytes: &[u8]) -> GPUMemory {
        let size_in_bytes = bytes.len() as vk::DeviceSize;

        unsafe {
            // Create buffer for raw bytes
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

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .expect_msg("Failed to create buffer");
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

            let memory = self
                .device
                .allocate_memory(&alloc_info, None)
                .expect_msg("Failed to allocate buffer memory");
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .expect_msg("Failed to bind buffer memory");

            // Map memory and write raw bytes
            let data_ptr = self
                .device
                .map_memory(memory, 0, size_in_bytes, vk::MemoryMapFlags::empty())
                .expect_msg("Failed to map buffer memory") as *mut u8;

            // Copy the data
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());

            self.device.unmap_memory(memory);

            GPUMemory::new(buffer, memory, size_in_bytes, self.device.clone())
        }
    }

    pub fn allocate_uninitialised_gpu_memory(&self, bytes: usize) -> Result<GPUMemory, VKMLError> {
        let size_in_bytes = bytes as vk::DeviceSize;

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

            Ok(GPUMemory::new(
                buffer,
                memory,
                size_in_bytes,
                self.device.clone(),
            ))
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

    fn create_pipeline(
        &self,
        shader_code: &[u8],
        local_size: [u32; 3],
    ) -> Result<vk::Pipeline, VKMLError> {
        unsafe {
            // ensure the shader byte length is a multiple of 4 (SPIR-V is in 32-bit words)
            if !shader_code.len().is_multiple_of(4) {
                return Err(VKMLError::Generic(
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
                layout: self.pipeline_layout,
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

    pub fn get_or_create_pipeline(&self, op: GPUOperation, local_size: [u32; 3]) -> vk::Pipeline {
        let key = (op, local_size);

        // Fast path: pipeline already exists
        if let Some(pipeline) = self.pipelines.read().unwrap().get(&key) {
            return *pipeline;
        }

        // Slow path: create pipeline
        let pipeline = self
            .create_pipeline(op.get_shader_bytes(), local_size)
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

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn extensions(&self) -> &VkExtensions {
        &self.extensions
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_descriptor_set_layout(&self) -> &vk::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn get_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn total_memory(&self) -> u64 {
        self.memory_tracker.get_maximum()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    pub fn has_compute(&self) -> bool {
        self.has_compute
    }

    pub fn max_workgroup_count(&self) -> [u32; 3] {
        self.max_workgroup_count
    }

    pub fn max_workgroup_size(&self) -> [u32; 3] {
        self.max_workgroup_size
    }

    pub fn max_workgroup_invocations(&self) -> u32 {
        self.max_workgroup_invocations
    }

    pub fn max_compute_queue_count(&self) -> u32 {
        self.max_compute_queue_count
    }

    pub fn max_shared_memory_size(&self) -> u32 {
        self.max_shared_memory_size
    }

    pub fn max_push_descriptors(&self) -> u32 {
        self.max_push_descriptors
    }

    pub fn get_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn get_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn get_or_create_timeline_semaphore(&self) -> vk::Semaphore {
        *self.timeline_semaphore.get_or_init(|| unsafe {
            let mut timeline_info = vk::SemaphoreTypeCreateInfo {
                s_type: vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO,
                next: ptr::null(),
                semaphore_type: vk::SemaphoreType::TIMELINE,
                initial_value: 0,
            };

            let semaphore_info = vk::SemaphoreCreateInfo {
                s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
                next: &mut timeline_info as *mut _ as *const c_void,
                flags: vk::SemaphoreCreateFlags::empty(),
            };

            self.device
                .create_semaphore(&semaphore_info, None)
                .expect("Failed to create timeline semaphore")
        })
    }

    /// Allocate the next N semaphore values and return the starting value
    pub fn allocate_semaphore_values(&self, count: u64) -> u64 {
        self.next_semaphore_value
            .fetch_add(count, Ordering::Relaxed)
    }

    /// Calculate optimal workgroup size for 1D compute operations (element-wise ops)
    pub fn optimal_workgroup_size_1d(&self, total_elements: u64) -> [u32; 3] {
        optimal_workgroup_size(
            self.max_workgroup_size,
            self.max_workgroup_invocations,
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
            self.max_workgroup_size,
            self.max_workgroup_invocations,
            [Some(rows), Some(cols), None],
        )
    }

    /// Calculate optimal workgroup size for 3D spatial operations (conv3d, maxpool3d)
    pub fn optimal_workgroup_size_3d(&self, x: u64, y: u64, z: u64) -> [u32; 3] {
        optimal_workgroup_size(
            self.max_workgroup_size,
            self.max_workgroup_invocations,
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

    /// Bind up to 4 GPU storage buffers to descriptor set bindings 0-3
    pub fn bind_storage_buffers(&self, command_buffer: vk::CommandBuffer, buffers: &[&GPUMemory]) {
        assert!(buffers.len() <= 4, "Maximum 4 buffers supported");

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
                self.get_layout(),
                0,
                &write_descriptor_sets,
            );
        }
    }

    /// Insert a memory barrier ensuring previous compute writes are visible to subsequent compute dispatches.
    pub fn barrier_compute_shader_access(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            let memory_barrier = vk::MemoryBarrier {
                s_type: vk::StructureType::MEMORY_BARRIER,
                next: ptr::null(),
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            };

            let buffer_barriers: [vk::BufferMemoryBarrier; 0] = [];
            let image_barriers: [vk::ImageMemoryBarrier; 0] = [];

            self.get_device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &buffer_barriers,
                &image_barriers,
            );
        }
    }

    /// Bind up to 4 GPU storage buffers (supporting Option<&GPUMemory>) to descriptor set bindings 0-3
    /// Optional buffers will be bound as null buffers with size 0
    pub fn bind_storage_buffers_optional(
        &self,
        command_buffer: vk::CommandBuffer,
        buffers: &[Option<&GPUMemory>],
    ) {
        assert!(buffers.len() <= 4, "Maximum 4 buffers supported");

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
                self.get_layout(),
                0,
                &write_descriptor_sets,
            );
        }
    }

    /// Bind a compute pipeline with the specified workgroup size
    pub fn bind_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        operation: GPUOperation,
        local_size: [u32; 3],
    ) {
        unsafe {
            let pipeline = self.get_or_create_pipeline(operation, local_size);
            self.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    /// Push constants to the compute shader
    pub fn bind_push_constants(&self, command_buffer: vk::CommandBuffer, data: &[u8]) {
        unsafe {
            self.get_device().cmd_push_constants(
                command_buffer,
                self.get_layout(),
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

    pub fn submit_with_timeline_semaphore(
        &self,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[(vk::Semaphore, u64)],
        signal_value: u64,
    ) -> Result<(), VKMLError> {
        unsafe {
            let timeline_sem = self.get_or_create_timeline_semaphore();

            let wait_sems: Vec<vk::Semaphore> = wait_semaphores.iter().map(|(s, _)| *s).collect();
            let wait_values: Vec<u64> = wait_semaphores.iter().map(|(_, v)| *v).collect();
            let signal_sems = [timeline_sem];
            let signal_values = [signal_value];

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo {
                s_type: vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO,
                next: ptr::null(),
                wait_semaphore_value_count: wait_values.len() as u32,
                wait_semaphore_values: if wait_values.is_empty() {
                    ptr::null()
                } else {
                    wait_values.as_ptr()
                },
                signal_semaphore_value_count: signal_values.len() as u32,
                signal_semaphore_values: signal_values.as_ptr(),
            };

            let wait_stages: Vec<vk::PipelineStageFlags> =
                vec![vk::PipelineStageFlags::COMPUTE_SHADER; wait_sems.len()];

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                next: &mut timeline_info as *mut _ as *const c_void,
                wait_semaphore_count: wait_sems.len() as u32,
                wait_semaphores: if wait_sems.is_empty() {
                    ptr::null()
                } else {
                    wait_sems.as_ptr()
                },
                wait_dst_stage_mask: if wait_stages.is_empty() {
                    ptr::null()
                } else {
                    wait_stages.as_ptr()
                },
                command_buffer_count: command_buffers.len() as u32,
                command_buffers: command_buffers.as_ptr(),
                signal_semaphore_count: signal_sems.len() as u32,
                signal_semaphores: signal_sems.as_ptr(),
            };

            self.device
                .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;

            Ok(())
        }
    }

    pub fn wait_for_timeline_value(&self, value: u64) -> Result<(), VKMLError> {
        unsafe {
            let timeline_sem = self.get_or_create_timeline_semaphore();
            let sems = [timeline_sem];
            let values = [value];

            let wait_info = vk::SemaphoreWaitInfo {
                s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                next: ptr::null(),
                flags: vk::SemaphoreWaitFlags::empty(),
                semaphore_count: 1,
                semaphores: sems.as_ptr(),
                values: values.as_ptr(),
            };

            self.device.wait_semaphores(&wait_info, u64::MAX)?;

            Ok(())
        }
    }
}
