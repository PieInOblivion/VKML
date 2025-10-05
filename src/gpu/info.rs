use std::ptr;

use crate::{
    error::VKMLError,
    gpu::{pool::GpuPool, raw_formats::RAW_FORMATS, vk_gpu::Gpu},
};

use vulkanalia::vk::{self, InstanceV1_0, InstanceV1_1};

#[derive(Clone, Debug)]
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
    max_push_descriptors: u32,
}

impl GpuInfo {
    pub fn new(gpu: &Gpu) -> GpuInfo {
        unsafe {
            let mut push_props = vk::PhysicalDevicePushDescriptorPropertiesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
                next: ptr::null_mut(),
                max_push_descriptors: 0,
            };

            let mut props2 = vk::PhysicalDeviceProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
                next: &mut push_props as *mut _ as *mut std::ffi::c_void,
                properties: Default::default(),
            };

            gpu.get_instance()
                .get_physical_device_properties2(gpu.get_physical_device(), &mut props2);

            let properties = props2.properties;

            let queue_families = gpu
                .get_instance()
                .get_physical_device_queue_family_properties(gpu.get_physical_device());

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
                        .get_instance()
                        .get_physical_device_format_properties(gpu.get_physical_device(), format);
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
                max_push_descriptors: push_props.max_push_descriptors,
            }
        }
    }

    pub fn system_gpus_info() -> Result<Vec<GpuInfo>, VKMLError> {
        let pool = GpuPool::new(None)?;

        let info: Vec<GpuInfo> = pool.gpus().iter().map(GpuInfo::new).collect();

        Ok(info)
    }
}
