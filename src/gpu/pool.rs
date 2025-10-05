use std::{collections::HashSet, ffi::CString, ptr, sync::Arc};

use vulkanalia::{
    Entry,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, InstanceV1_0},
};

use crate::{GpuInfo, error::VKMLError, gpu::vk_gpu::Gpu, utils::expect_msg::ExpectMsg};

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

impl std::fmt::Debug for GpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let infos: Vec<GpuInfo> = self.gpus.iter().map(|g| g.get_info()).collect();
        f.debug_struct("GpuPool")
            .field("gpus_info", &infos)
            .finish()
    }
}
