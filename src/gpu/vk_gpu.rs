use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    ptr,
    sync::{Arc, RwLock},
};
use vulkanalia::{
    Device, Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, DeviceV1_0, Handle, InstanceV1_0},
};

use crate::{
    compute::memory_tracker::MemoryTracker,
    gpu::raw_formats::RAW_FORMATS,
    instruction::gpu_operations::GPUMemoryOperation,
    utils::{error::VKMLError, expect_msg::ExpectMsg},
};

use super::gpu_memory::GPUMemory;
use super::vk_extensions::VkExtensions;

const REQUEST_ALL_COMPUTE_QUEUES: bool = false;

pub struct GpuPool {
    gpus: Vec<Gpu>,
    _entry: Entry,
}

impl GpuPool {
    pub fn new(selected: Option<Vec<usize>>) -> Result<Self, VKMLError> {
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY).expect_msg("Failed to load Vulkan library");
            let entry = Entry::new(loader).expect_msg("Failed to create Vulkan entry point");

            let aname = CString::new("vkml").unwrap();

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

            let instance = Arc::new(entry.create_instance(&create_info, None)?);

            // Get all gpus
            let physical_devices = instance.enumerate_physical_devices()?;

            let mut init_gpus = Vec::new();

            // If selected is Some, iterate over those indices and validate them.
            // Otherwise initialise every physical device found.
            if let Some(selected_set) = selected {
                let mut seen = HashSet::new();

                for &idx in selected_set.iter() {
                    if idx >= physical_devices.len() {
                        return Err(VKMLError::Generic(format!(
                            "Selected GPU index {} out of range",
                            idx
                        )));
                    }

                    if !seen.insert(idx) {
                        return Err(VKMLError::Generic(format!(
                            "Duplicate GPU index {} in selection",
                            idx
                        )));
                    }

                    let gpu = Gpu::new_shared(instance.clone(), physical_devices[idx])?;
                    init_gpus.push(gpu);
                }
            } else {
                for (idx, _) in physical_devices.iter().enumerate() {
                    let gpu = Gpu::new_shared(instance.clone(), physical_devices[idx])?;
                    init_gpus.push(gpu);
                }
            }

            let gpus = Self {
                gpus: init_gpus,
                _entry: entry,
            };

            Ok(gpus)
        }
    }

    pub fn gpus(&self) -> &Vec<Gpu> {
        &self.gpus
    }

    pub fn get_gpu(&self, idx: usize) -> &Gpu {
        self.gpus().get(idx).unwrap()
    }
}

// NOTE: Can pipelines be a vec of oncelock?

pub struct Gpu {
    physical_device: vk::PhysicalDevice,
    compute_queues: Vec<vk::Queue>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    memory_tracker: MemoryTracker,
    extensions: VkExtensions,

    // These must be dropped in this order
    pipelines: RwLock<HashMap<GPUMemoryOperation, vk::Pipeline>>,
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

            // Get queue family properties to find the number of available queues
            let queue_family_props =
                instance.get_physical_device_queue_family_properties(physical_device);

            let queue_count = queue_family_props[queue_family_index as usize].queue_count;
            let requested_queue_count: u32 = if REQUEST_ALL_COMPUTE_QUEUES {
                queue_count
            } else {
                1
            };

            // Create queue info with priorities for all queues
            let queue_priorities = vec![1.0; requested_queue_count as usize];

            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count: requested_queue_count,
                queue_priorities: queue_priorities.as_ptr(),
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

            // Get all compute queues
            let mut compute_queues = Vec::with_capacity(requested_queue_count as usize);
            for i in 0..requested_queue_count {
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
                // Additional bindings if needed (bias buffer for Conv, etc.)
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
                physical_device,
                compute_queues,
                descriptor_set_layout,
                memory_tracker: MemoryTracker::new((total_memory as f64 * 0.6) as u64), // TODO: 60%, kept low for testing
                extensions: vk_extensions,

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

    pub fn submit_command_buffers_and_wait(
        &self,
        command_buffers: &[vk::CommandBuffer],
    ) -> Result<(), VKMLError> {
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
            self.device.wait_for_fences(&fences, true, std::u64::MAX)?;

            for fence in fences {
                self.device.destroy_fence(fence, None);
            }

            Ok(())
        }
    }

    fn create_pipeline(&self, shader_code: &[u8]) -> Result<vk::Pipeline, VKMLError> {
        unsafe {
            let aligned_code: Vec<u32>;
            if shader_code.as_ptr().align_offset(4) != 0 {
                let mut padded = Vec::with_capacity((shader_code.len() + 3) / 4 * 4);
                padded.extend_from_slice(shader_code);
                while padded.len() % 4 != 0 {
                    padded.push(0);
                }
                aligned_code = padded
                    .chunks_exact(4)
                    .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
            } else {
                // shader_code is 4-byte aligned; convert bytes -> u32 words safely
                aligned_code = shader_code
                    .chunks_exact(4)
                    .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
            }

            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::ShaderModuleCreateFlags::empty(),
                code_size: aligned_code.len() * 4,
                code: aligned_code.as_ptr(),
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

    fn create_and_get_pipeline(&self, op: GPUMemoryOperation, shader_code: &[u8]) -> vk::Pipeline {
        let mut compute_pipeline_write = self.pipelines.write().unwrap();

        let new_pipeline = self
            .create_pipeline(shader_code)
            .expect_msg(&format!("Pipeline creation failed {:?}", op));

        compute_pipeline_write.insert(op, new_pipeline);

        new_pipeline
    }

    fn get_pipeline_for_op(&self, op: GPUMemoryOperation) -> Option<vk::Pipeline> {
        self.pipelines.read().unwrap().get(&op).copied()
    }

    pub fn get_or_create_pipeline(&self, op: GPUMemoryOperation) -> vk::Pipeline {
        if let Some(pipeline) = self.get_pipeline_for_op(op) {
            return pipeline;
        } else {
            return self.create_and_get_pipeline(op, op.get_shader_bytes());
        }
    }

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn extensions(&self) -> &VkExtensions {
        &self.extensions
    }

    pub fn get_device(&self) -> &Device {
        // Arc<Device> derefs to Device, return a reference to the inner Device
        &*self.device
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

    pub fn get_info(&self) -> GpuInfo {
        GpuInfo::new(&self)
    }
}

pub struct GpuInfo {
    name: String,
    device_type: vk::PhysicalDeviceType,
    has_compute: bool,
    supported_formats: Vec<vk::Format>,
    max_workgroup_count: [u32; 3],
    max_workgroup_size: [u32; 3],
    max_workgroup_invocations: u32,
    max_shared_memory_size: u32,
    compute_queue_count: u32,
}

impl GpuInfo {
    fn new(gpu: &Gpu) -> GpuInfo {
        unsafe {
            let properties = gpu
                .instance
                .get_physical_device_properties(gpu.physical_device);
            let queue_families = gpu
                .instance
                .get_physical_device_queue_family_properties(gpu.physical_device);

            let name = String::from_utf8_lossy(
                &properties
                    .device_name
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect::<Vec<u8>>(),
            )
            .to_string();

            let (has_compute, compute_queue_count) = queue_families
                .iter()
                .find(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|props| (true, props.queue_count))
                .unwrap_or((false, 0));

            let supported_formats = RAW_FORMATS
                .iter()
                .cloned()
                .filter(|&format| {
                    let props = gpu
                        .instance
                        .get_physical_device_format_properties(gpu.physical_device, format);
                    props
                        .buffer_features
                        .contains(vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER)
                })
                .collect();

            GpuInfo {
                name,
                device_type: properties.device_type,
                has_compute,
                supported_formats,
                max_workgroup_count: properties.limits.max_compute_work_group_count,
                max_workgroup_size: properties.limits.max_compute_work_group_size,
                max_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
                max_shared_memory_size: properties.limits.max_compute_shared_memory_size,
                compute_queue_count,
            }
        }
    }

    pub fn system_gpus_info() -> Result<Vec<GpuInfo>, VKMLError> {
        let pool = GpuPool::new(None)?;

        let info: Vec<GpuInfo> = pool.gpus().iter().map(|gpu| GpuInfo::new(gpu)).collect();

        Ok(info)
    }
}
