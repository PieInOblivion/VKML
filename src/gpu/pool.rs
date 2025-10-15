use std::{collections::HashSet, ffi::CString, ptr, sync::Arc};

use vulkanalia::{
    Entry,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, InstanceV1_0},
};

use crate::{error::VKMLError, gpu::vk_gpu::Gpu, utils::expect_msg::ExpectMsg};

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
                application_version: vk::make_version(1, 3, 0),
                engine_name: aname.as_ptr(),
                engine_version: vk::make_version(1, 3, 0),
                api_version: vk::make_version(1, 3, 0),
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

            let physical_devices = instance.enumerate_physical_devices()?;

            let mut init_gpus = Vec::new();

            // If selected is Some, iterate over those indices and validate them.
            // Otherwise initialise every physical device found.
            if let Some(selected_set) = selected {
                init_gpus.reserve_exact(selected_set.len());
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

                    init_gpus.push(Gpu::new_shared(instance.clone(), physical_devices[idx])?);
                }
            } else {
                init_gpus.reserve_exact(physical_devices.len());
                for device in physical_devices {
                    init_gpus.push(Gpu::new_shared(instance.clone(), device)?);
                }

                // Sort GPUs: discrete GPUs first, then by total memory (descending)
                init_gpus.sort_by_key(|gpu| {
                    (
                        gpu.device_type() != vk::PhysicalDeviceType::DISCRETE_GPU,
                        std::cmp::Reverse(gpu.total_memory()),
                    )
                });
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

impl std::fmt::Debug for GpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpu_debugs: Vec<String> = self
            .gpus
            .iter()
            .map(|g| {
                format!(
                    "{{ name: {:?}, device_type: {:?}, has_compute: {}, max_workgroup_count: {:?}, max_workgroup_size: {:?}, max_workgroup_invocations: {}, max_compute_queue_count: {}, max_shared_memory_size: {}, max_push_descriptors: {}, coop_matrix: {:?} }}",
                    g.name(),
                    g.device_type(),
                    g.has_compute(),
                    g.max_workgroup_count(),
                    g.max_workgroup_size(),
                    g.max_workgroup_invocations(),
                    g.max_compute_queue_count(),
                    g.max_shared_memory_size(),
                    g.max_push_descriptors(),
                    g.extensions().coop_matrix_shapes(),
                )
            })
            .collect();

        f.debug_struct("GpuPool")
            .field("gpus", &gpu_debugs)
            .finish()
    }
}
