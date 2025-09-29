use crate::utils::error::VKMLError;
use std::any::Any;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;
use vulkanalia::vk;

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
    pub cooperative_matrix: bool,
    pub shader_float16_int8: bool,
    pub timeline_semaphore: bool,
    pub synchronization2: bool,
    pub memory_budget: bool,
    pub push_descriptor: bool,
}

impl VkExtensions {
    // extension names we care about
    pub const VK_KHR_COOPERATIVE_MATRIX: &'static str = "VK_KHR_cooperative_matrix";
    pub const VK_KHR_SHADER_FLOAT16_INT8: &'static str = "VK_KHR_shader_float16_int8";
    pub const VK_KHR_TIMELINE_SEMAPHORE: &'static str = "VK_KHR_timeline_semaphore";
    pub const VK_KHR_SYNCHRONIZATION2: &'static str = "VK_KHR_synchronization2";
    pub const VK_EXT_MEMORY_BUDGET: &'static str = "VK_EXT_memory_budget";
    pub const VK_KHR_PUSH_DESCRIPTOR: &'static str = "VK_KHR_push_descriptor";

    // build from vk::ExtensionProperties returned by Vulkan
    pub fn from_extension_properties(props: &[vk::ExtensionProperties]) -> Result<Self, VKMLError> {
        let mut res = VkExtensions::default();

        for p in props {
            // convert fixed-size name array to string
            let name_cstr = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
            let name = name_cstr.to_string_lossy();

            match name.as_ref() {
                Self::VK_KHR_COOPERATIVE_MATRIX => res.cooperative_matrix = true,
                Self::VK_KHR_SHADER_FLOAT16_INT8 => res.shader_float16_int8 = true,
                Self::VK_KHR_TIMELINE_SEMAPHORE => res.timeline_semaphore = true,
                Self::VK_KHR_SYNCHRONIZATION2 => res.synchronization2 = true,
                Self::VK_EXT_MEMORY_BUDGET => res.memory_budget = true,
                Self::VK_KHR_PUSH_DESCRIPTOR => res.push_descriptor = true,
                _ => {}
            }
        }

        if !res.push_descriptor {
            return Err(VKMLError::Generic(
                "Required device extension VK_KHR_push_descriptor not present".to_string(),
            ));
        }

        Ok(res)
    }

    // return owned CStrings for extensions we want to enable
    pub fn enabled_extension_names(&self) -> Vec<CString> {
        let mut v = Vec::new();

        if self.cooperative_matrix {
            v.push(CString::new(Self::VK_KHR_COOPERATIVE_MATRIX).unwrap());
        }
        if self.shader_float16_int8 {
            v.push(CString::new(Self::VK_KHR_SHADER_FLOAT16_INT8).unwrap());
        }
        if self.timeline_semaphore {
            v.push(CString::new(Self::VK_KHR_TIMELINE_SEMAPHORE).unwrap());
        }
        if self.synchronization2 {
            v.push(CString::new(Self::VK_KHR_SYNCHRONIZATION2).unwrap());
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
        // if no feature structs needed, return an empty pnext owner
        if !self.timeline_semaphore
            && !self.synchronization2
            && !self.shader_float16_int8
            && !self.cooperative_matrix
            && !self.memory_budget
        {
            return DeviceCreateExtras {
                names,
                name_ptrs,
                pnext: DevicePNext {
                    ptr: ptr::null_mut(),
                    _holders: Vec::new(),
                },
            };
        }

        // build a p_next chain: box each feature struct and link via next
        let mut holders: Vec<Box<dyn Any>> = Vec::new();
        let mut head: *mut c_void = ptr::null_mut();

        if self.cooperative_matrix {
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

        if self.shader_float16_int8 {
            let mut feat = Box::new(vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                next: ptr::null_mut(),
                shader_float16: vk::TRUE,
                shader_int8: vk::TRUE,
                ..Default::default()
            });
            feat.next = head as *mut _;
            head = (&*feat) as *const _ as *mut c_void;
            holders.push(feat);
        }

        if self.timeline_semaphore {
            let mut feat = Box::new(vk::PhysicalDeviceTimelineSemaphoreFeatures {
                s_type: vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
                next: ptr::null_mut(),
                timeline_semaphore: vk::TRUE,
                ..Default::default()
            });
            feat.next = head as *mut _;
            head = (&*feat) as *const _ as *mut c_void;
            holders.push(feat);
        }

        if self.synchronization2 {
            let mut feat = Box::new(vk::PhysicalDeviceSynchronization2Features {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                next: ptr::null_mut(),
                synchronization2: vk::TRUE,
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
}
