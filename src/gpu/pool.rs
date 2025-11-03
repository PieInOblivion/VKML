use std::{collections::HashSet, ffi::CString, ptr, sync::Arc};

use vulkanalia::{
    Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, InstanceV1_0},
};
use zero_pool::{global_pool, zp_define_task_fn};

use crate::{error::VKMLError, gpu::vk_gpu::Gpu, utils::expect_msg::ExpectMsg};

pub struct GpuPool {
    gpus: Vec<Arc<Gpu>>,
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

            // If selected is Some, iterate over those indices and validate them.
            // Otherwise initialise every physical device found.
            let init_gpus = if let Some(selected_set) = selected {
                // validate all indices before spawning any tasks
                let mut seen = HashSet::new();
                let mut validated_indices = Vec::with_capacity(selected_set.len());

                for &idx in selected_set.iter() {
                    if idx >= physical_devices.len() {
                        return Err(VKMLError::GpuPool(format!(
                            "Selected GPU index {} out of range",
                            idx
                        )));
                    }

                    if !seen.insert(idx) {
                        return Err(VKMLError::GpuPool(format!(
                            "Duplicate GPU index {} in selection",
                            idx
                        )));
                    }

                    validated_indices.push(idx);
                }

                let mut gpus = Vec::with_capacity(validated_indices.len());

                let tasks: Vec<GpuInitParams> = validated_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| GpuInitParams {
                        instance: instance.clone(),
                        physical_device: physical_devices[idx],
                        index: i,
                        out_ptr: gpus.as_mut_ptr(),
                    })
                    .collect();

                global_pool()
                    .submit_batch_uniform(gpu_init_task, &tasks)
                    .wait();

                gpus.set_len(validated_indices.len());

                gpus
            } else {
                let count = physical_devices.len();
                let mut gpus = Vec::with_capacity(count);

                let tasks: Vec<GpuInitParams> = physical_devices
                    .iter()
                    .enumerate()
                    .map(|(i, &physical_device)| GpuInitParams {
                        instance: instance.clone(),
                        physical_device,
                        index: i,
                        out_ptr: gpus.as_mut_ptr(),
                    })
                    .collect();

                global_pool()
                    .submit_batch_uniform(gpu_init_task, &tasks)
                    .wait();

                gpus.set_len(count);

                // Sort GPUs: discrete GPUs first, then by total memory (descending)
                gpus.sort_by_key(|gpu| {
                    (
                        gpu.device_type() != vk::PhysicalDeviceType::DISCRETE_GPU,
                        std::cmp::Reverse(gpu.memory_total()),
                    )
                });

                gpus
            };

            let gpus = Self {
                gpus: init_gpus,
                _entry: entry,
            };

            Ok(gpus)
        }
    }

    pub fn gpus(&self) -> &Vec<Arc<Gpu>> {
        &self.gpus
    }

    pub fn get_gpu(&self, idx: usize) -> Arc<Gpu> {
        self.gpus()
            .get(idx)
            .cloned()
            .unwrap_or_else(|| panic!("Requested GPU index {idx} out of range"))
    }
}

impl std::fmt::Debug for GpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpu_debugs: Vec<String> = self
            .gpus
            .iter()
            .map(|g| {
                let staging_desc = match g.staging_buffer_info() {
                    Some((size, props)) => format!(
                        "Some {{ size_bytes: {}, properties: {:?} }}",
                        size, props
                    ),
                    None => "None".to_string(),
                };
                format!(
                    "{{ name: `{}`, device_type: {:?}, has_compute: {}, memory_budget: {}, memory_in_use: {}, memory_in_use_as_percent: {:.2}%, max_workgroup_count: {:?}, max_workgroup_size: {:?}, max_workgroup_invocations: {}, max_compute_queue_count: {}, max_shared_memory_size: {}, max_push_descriptors: {}, subgroup_size: {}, host_visible_device_local_bytes: {}, host_access_mode: {:?}, staging_buffer: {}, coop_matrix: {:?} }}",
                    g.name(),
                    g.device_type(),
                    g.has_compute(),
                    g.memory_available(),
                    g.memory_current(),
                    if g.memory_total() == 0 {
                        0.0
                    } else {
                        (g.memory_current() as f64 / g.memory_total() as f64) * 100.0
                    },
                    g.max_workgroup_count(),
                    g.max_workgroup_size(),
                    g.max_workgroup_invocations(),
                    g.max_compute_queue_count(),
                    g.max_shared_memory_size(),
                    g.max_push_descriptors(),
                    g.subgroup_size(),
                    g.host_visible_device_local_bytes(),
                    g.host_access_mode(),
                    staging_desc,
                    g.extensions().coop_matrix_shapes(),
                )
            })
            .collect();

        f.debug_struct("GpuPool")
            .field("gpus", &gpu_debugs)
            .finish()
    }
}

struct GpuInitParams {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    index: usize,
    out_ptr: *mut Arc<Gpu>,
}

zp_define_task_fn!(gpu_init_task, GpuInitParams, |params| {
    let gpu = Arc::new(
        Gpu::new_shared(params.instance.clone(), params.physical_device)
            .expect_msg("Failed to initialize GPU"),
    );

    unsafe {
        let slot = params.out_ptr.add(params.index);
        ptr::write(slot, gpu);
    }
});
