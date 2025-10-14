use crate::utils::error::VKMLError;
use std::any::Any;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;
use vulkanalia::vk::InstanceV1_0;
use vulkanalia::{Instance, vk};

// helpers for preparing device extension names and an owned p_next chain
// Returned value owns CStrings and any boxed feature structs. keep it
// alive until after create_device returns so pointers stay valid.
pub struct DeviceCreateExtras {
    // owned extension names must be kept alive while name_ptrs are used
    pub names: Vec<CString>,
    // raw pointers into names suitable for passing to Vulkan create info
    pub name_ptrs: Vec<*const c_char>,
    // owner for a heap-allocated p_next chain. Use pnext.ptr as the
    // DeviceCreateInfo::next value. When no features are enabled pnext.ptr
    // will be null and _holders will be empty.
    pub pnext: DevicePNext,
}

impl DeviceCreateExtras {
    // return a *const c_void for DeviceCreateInfo::next
    // caller must keep this struct alive while the pointer is used
    pub fn device_create_next(&self) -> *const std::ffi::c_void {
        self.pnext.ptr as *const std::ffi::c_void
    }
}

// owner for a heap-allocated p_next chain. ptr is the head; _holders
// keeps the boxed structs alive while this owner is in scope.
pub struct DevicePNext {
    pub ptr: *mut std::ffi::c_void,
    // hold boxed structs type-erased so they drop when this struct is dropped
    _holders: Vec<Box<dyn Any>>,
}

#[derive(Clone, Debug, Default)]
pub struct VkExtensions {
    cooperative_matrix: Option<(u32, u32, u32)>, // (max_workgroup_size_x, max_workgroup_size_y, max_workgroup_size_z)
    memory_budget: bool,
    push_descriptor: bool,
}

impl VkExtensions {
    // extension names we care about
    pub const VK_KHR_COOPERATIVE_MATRIX: &'static str = "VK_KHR_cooperative_matrix";
    pub const VK_EXT_MEMORY_BUDGET: &'static str = "VK_EXT_memory_budget";
    pub const VK_KHR_PUSH_DESCRIPTOR: &'static str = "VK_KHR_push_descriptor";

    pub fn from_extension_properties(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        props: &[vk::ExtensionProperties],
    ) -> Result<Self, VKMLError> {
        let mut res = VkExtensions::default();

        for p in props {
            // convert fixed-size name array to string
            let name_cstr = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
            let name = name_cstr.to_string_lossy();

            match name.as_ref() {
                Self::VK_KHR_COOPERATIVE_MATRIX => {
                    res.cooperative_matrix =
                        VkExtensions::query_cooperative_matrix_limits(instance, physical_device);
                }
                Self::VK_EXT_MEMORY_BUDGET => res.memory_budget = true,
                Self::VK_KHR_PUSH_DESCRIPTOR => res.push_descriptor = true,
                _ => {}
            }
        }
        Ok(res)
    }

    fn query_cooperative_matrix_limits(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<(u32, u32, u32)> {
        unsafe {
            let mut count: u32 = 0;
            let fp = (*instance)
                .commands()
                .get_physical_device_cooperative_matrix_properties_khr;
            // first call to get count
            fp(physical_device, &mut count, ptr::null_mut());

            if count == 0 {
                // extension present but no entries reported so return zeroed limits. Maybe we should return none?
                return Some((0, 0, 0));
            }

            let mut properties: Vec<vk::CooperativeMatrixPropertiesKHR> =
                Vec::with_capacity(count as usize);
            let result = fp(physical_device, &mut count, properties.as_mut_ptr());
            if result != vk::Result::SUCCESS {
                return Some((0, 0, 0));
            }
            properties.set_len(count as usize);

            let mut max_m = 0u32;
            let mut max_n = 0u32;
            let mut max_k = 0u32;
            for p in properties.iter() {
                max_m = max_m.max(p.m_size);
                max_n = max_n.max(p.n_size);
                max_k = max_k.max(p.k_size);
            }

            Some((max_m, max_n, max_k))
        }
    }

    // return owned CStrings for extensions we want to enable
    pub fn enabled_extension_names(&self) -> Vec<CString> {
        let mut v = Vec::new();

        if self.cooperative_matrix.is_some() {
            v.push(CString::new(Self::VK_KHR_COOPERATIVE_MATRIX).unwrap());
        }
        if self.memory_budget {
            v.push(CString::new(Self::VK_EXT_MEMORY_BUDGET).unwrap());
        }
        if self.push_descriptor {
            v.push(CString::new(Self::VK_KHR_PUSH_DESCRIPTOR).unwrap());
        }

        v
    }

    // prepare CStrings and an owned p_next chain (if needed)
    // returned struct owns everything; keep it alive through create_device
    pub fn prepare_device_create(&self) -> DeviceCreateExtras {
        let names = self.enabled_extension_names();
        let name_ptrs: Vec<*const c_char> =
            names.iter().map(|s| s.as_ptr() as *const c_char).collect();

        let mut holders: Vec<Box<dyn Any>> = Vec::new();
        let mut head: *mut c_void = ptr::null_mut();

        if self.cooperative_matrix.is_some() {
            // enable the cooperative matrix feature flag in the p_next chain
            let mut feat = Box::new(vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                next: ptr::null_mut(),
                cooperative_matrix: vk::TRUE,
                ..Default::default()
            });
            feat.next = head as *mut _;
            head = (&*feat) as *const _ as *mut c_void;
            holders.push(feat);
        }

        DeviceCreateExtras {
            names,
            name_ptrs,
            pnext: DevicePNext {
                ptr: head,
                _holders: holders,
            },
        }
    }

    pub fn coop_matrix_limits(&self) -> Option<(u32, u32, u32)> {
        self.cooperative_matrix
    }
}
