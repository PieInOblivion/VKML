use vk::Format;
use vulkanalia::{Instance, vk, vk::InstanceV1_0};

/// Simplified GPU information structure
#[derive(Debug)]
pub struct GPUInfo {
    pub name: String,
    pub device_index: usize,
    pub total_memory: u64,
    pub device_type: vk::PhysicalDeviceType,
    pub has_compute: bool,
    pub vk_ext_memory_budget: bool,
    pub supported_formats: Vec<vk::Format>,
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_shared_memory_size: u32,
    pub compute_queue_count: u32,
}

impl GPUInfo {
    pub fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device_index: usize,
    ) -> Self {
        // List of raw numerical formats we want to support
        // TODO: This probably has a better place to exist
        let raw_formats = [
            Format::R8_UINT,
            Format::R8_SINT,
            Format::R16_UINT,
            Format::R16_SINT,
            Format::R16_SFLOAT,
            Format::R32_UINT,
            Format::R32_SINT,
            Format::R32_SFLOAT,
            Format::R64_UINT,
            Format::R64_SINT,
            Format::R64_SFLOAT,
        ];

        unsafe {
            let properties = instance.get_physical_device_properties(physical_device);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let name = String::from_utf8_lossy(
                &properties
                    .device_name
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect::<Vec<u8>>(),
            )
            .to_string();

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

            let (has_compute, compute_queue_count) = queue_families
                .iter()
                .find(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|props| (true, props.queue_count))
                .unwrap_or((false, 0));

            let vk_ext_memory_budget = instance
                .enumerate_device_extension_properties(physical_device, None)
                .map(|extensions| {
                    extensions.iter().any(|ext| {
                        let name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr());
                        name.to_bytes() == b"VK_EXT_memory_budget"
                    })
                })
                .unwrap_or(false);

            let supported_formats = raw_formats
                .iter()
                .cloned()
                .filter(|&format| {
                    let props =
                        instance.get_physical_device_format_properties(physical_device, format);
                    props
                        .buffer_features
                        .contains(vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER)
                })
                .collect();

            Self {
                name,
                device_index,
                total_memory,
                device_type: properties.device_type,
                has_compute,
                vk_ext_memory_budget,
                supported_formats,
                max_workgroup_count: properties.limits.max_compute_work_group_count,
                max_workgroup_size: properties.limits.max_compute_work_group_size,
                max_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
                max_shared_memory_size: properties.limits.max_compute_shared_memory_size,
                compute_queue_count,
            }
        }
    }
}
