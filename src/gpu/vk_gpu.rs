use std::{
    array,
    ffi::{CString, c_void},
    ptr,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use vulkanalia::{
    Device, Instance,
    vk::{
        self, DeviceV1_0, DeviceV1_2, Handle, InstanceV1_0, InstanceV1_1,
        KhrPushDescriptorExtension,
    },
};

use crate::{
    compute::memory_tracker::MemoryTracker,
    instruction::gpu_operations::GPUOperation,
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

    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    descriptor_set_layout: vk::DescriptorSetLayout,
    memory_tracker: MemoryTracker,
    extensions: VkExtensions,

    // Drop order matters: fields drop top-to-bottom
    timeline_semaphore: OnceLock<vk::Semaphore>,
    next_semaphore_value: AtomicU64,
    pipelines: Vec<OnceLock<vk::Pipeline>>,
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
            let vk_extensions = VkExtensions::from_extension_properties(&device_extensions)?;

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

            // Query device properties for info fields and limits
            let mut push_props = vk::PhysicalDevicePushDescriptorPropertiesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
                next: ptr::null_mut(),
                max_push_descriptors: 0,
            };

            let mut props2 = vk::PhysicalDeviceProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
                next: &mut push_props as *mut _ as *mut c_void,
                properties: Default::default(),
            };

            instance.get_physical_device_properties2(physical_device, &mut props2);
            let properties = props2.properties;

            // Extract device name
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

            let pipelines = (0..GPUOperation::VARIANT_COUNT)
                .map(|_| OnceLock::new())
                .collect();

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

                physical_device,
                compute_queue,
                descriptor_set_layout,
                memory_tracker: MemoryTracker::new((total_memory as f64 * 0.6) as u64), // TODO: 60%, kept low for testing
                extensions: vk_extensions,
                timeline_semaphore: OnceLock::new(),
                next_semaphore_value: AtomicU64::new(1),

                pipelines,
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

    fn create_pipeline(&self, shader_code: &[u8]) -> Result<vk::Pipeline, VKMLError> {
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
                    specialization_info: std::ptr::null(),
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

    fn create_and_get_pipeline(&self, op: GPUOperation, shader_code: &[u8]) -> vk::Pipeline {
        let pipeline_ref = self.pipelines[op as usize].get_or_init(|| {
            self.create_pipeline(shader_code)
                .expect_msg(&format!("Pipeline creation failed {:?}", op))
        });

        *pipeline_ref
    }

    fn get_pipeline_for_op(&self, op: GPUOperation) -> Option<vk::Pipeline> {
        self.pipelines[op as usize].get().copied()
    }

    pub fn get_or_create_pipeline(&self, op: GPUOperation) -> vk::Pipeline {
        if let Some(pipeline) = self.get_pipeline_for_op(op) {
            pipeline
        } else {
            self.create_and_get_pipeline(op, op.get_shader_bytes())
        }
    }

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn extensions(&self) -> &VkExtensions {
        &self.extensions
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
    /// Counts in increments of 64 up to either total_elements or GPU limits
    pub fn optimal_workgroup_size_1d(&self, total_elements: u64) -> u32 {
        const INCREMENT: u32 = 64;
        let max_total = self.max_workgroup_invocations;
        let max_x = self.max_workgroup_size[0];

        let mut size = INCREMENT;
        while size < max_total && size < max_x && (size as u64) < total_elements {
            size += INCREMENT;
        }

        size.min(max_total).min(max_x)
    }

    /// Calculate optimal workgroup size for 2D compute operations (matmul, conv2d)
    /// Returns square tiles, counting in increments of 8 per dimension
    ///
    /// Note: For batched 2D operations (e.g., batched matmul), use this function
    /// and handle the batch dimension in the dispatch call, not the workgroup size.
    /// Standard pattern: workgroup = [tile, tile], dispatch = [m/tile, n/tile, batch]
    pub fn optimal_workgroup_size_2d(&self, rows: u64, cols: u64) -> [u32; 2] {
        const INCREMENT: u32 = 8;
        let max_total = self.max_workgroup_invocations;
        let max_dim_gpu = self.max_workgroup_size[0].min(self.max_workgroup_size[1]);
        let max_dim_op = rows.max(cols);

        let mut tile_size = INCREMENT;
        while (tile_size + INCREMENT).pow(2) <= max_total
            && (tile_size + INCREMENT) <= max_dim_gpu
            && (tile_size + INCREMENT) as u64 <= max_dim_op
        {
            tile_size += INCREMENT;
        }

        [tile_size, tile_size]
    }

    /// Calculate optimal workgroup size for 3D spatial operations (conv3d, maxpool3d)
    /// Returns cubic tiles, counting in increments of 2 per dimension
    ///
    /// This is for true 3D spatial operations, not batched 2D operations.
    /// For batched 2D, use optimal_workgroup_size_2d() instead.
    ///
    /// Uses the minimum of max_workgroup_size dimensions to ensure cubic tiles fit.
    pub fn optimal_workgroup_size_3d(&self, x: u64, y: u64, z: u64) -> [u32; 3] {
        const INCREMENT: u32 = 2;
        let max_total = self.max_workgroup_invocations;
        let max_dim_gpu = self.max_workgroup_size[0]
            .min(self.max_workgroup_size[1])
            .min(self.max_workgroup_size[2]);
        let max_dim_op = x.max(y).max(z);

        let mut size = INCREMENT;
        while (size + INCREMENT).pow(3) <= max_total
            && (size + INCREMENT) <= max_dim_gpu
            && (size + INCREMENT) as u64 <= max_dim_op
        {
            size += INCREMENT;
        }

        [size, size, size]
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
    ) {
        unsafe {
            let pipeline = self.get_or_create_pipeline(operation);
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
    pub fn dispatch(&self, command_buffer: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        unsafe {
            self.get_device().cmd_dispatch(command_buffer, x, y, z);
        }
    }

    /// End command buffer recording
    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<(), VKMLError> {
        unsafe {
            self.get_device().end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    // TODO: To use these workgroup sizes in shaders:
    //
    // 1. Update shaders to use specialization constants instead of hardcoded sizes:
    //    Change:
    //      layout(local_size_x = 256) in;
    //    To:
    //      layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
    //
    // 2. When creating pipelines (in get_or_create_pipeline), pass specialization constants:
    //      let workgroup_size = gpu.optimal_workgroup_size_1d(total_elements);
    //      let spec_data = [workgroup_size, 1u32, 1u32];
    //      let spec_entries = [
    //          vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 },
    //          vk::SpecializationMapEntry { constant_id: 1, offset: 4, size: 4 },
    //          vk::SpecializationMapEntry { constant_id: 2, offset: 8, size: 4 },
    //      ];
    //      let spec_info = vk::SpecializationInfo {
    //          map_entry_count: 3,
    //          p_map_entries: spec_entries.as_ptr(),
    //          data_size: 12,
    //          p_data: spec_data.as_ptr() as *const c_void,
    //      };
    //      // Pass spec_info to vk::ComputePipelineCreateInfo.stage.specialization_info
    //
    // 3. Update dispatch calculations in create_command_buffer():
    //      let workgroup_size = gpu.optimal_workgroup_size_1d(total_elements);
    //      let num_workgroups = total_elements.div_ceil(workgroup_size as u64);
    //      gpu.get_device().cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

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
            let signal_sems = vec![timeline_sem];
            let signal_values = vec![signal_value];

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
