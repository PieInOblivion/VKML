use crate::utils::error::VKMLError;
use crate::utils::vk_to_onnx_dtype::vk_to_onnx_dtype;
use onnx_extractor::DataType;
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
pub struct CoopMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: DataType,
    pub b_type: DataType,
    pub c_type: DataType,
    pub result_type: DataType,
}

#[derive(Debug, Default)]
pub struct VkExtensions {
    memory_budget: bool,
    shader_float_16_int8: bool,
    storage_16bit: bool,
    shader_bfloat16: bool,
    cooperative_matrix: Option<Vec<CoopMatrixShape>>,
}

impl VkExtensions {
    // extension names we care about
    pub const VK_KHR_COOPERATIVE_MATRIX: &'static str = "VK_KHR_cooperative_matrix";
    pub const VK_EXT_MEMORY_BUDGET: &'static str = "VK_EXT_memory_budget";
    pub const VK_KHR_SHADER_FLOAT16_INT8: &'static str = "VK_KHR_shader_float16_int8";
    pub const VK_KHR_16BIT_STORAGE: &'static str = "VK_KHR_16bit_storage";
    pub const VK_KHR_SHADER_BFLOAT16: &'static str = "VK_KHR_shader_bfloat16";

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
                Self::VK_KHR_SHADER_FLOAT16_INT8 => {
                    res.shader_float_16_int8 = true;
                }
                Self::VK_KHR_16BIT_STORAGE => {
                    res.storage_16bit = true;
                }
                Self::VK_KHR_SHADER_BFLOAT16 => {
                    res.shader_bfloat16 = true;
                }
                _ => {}
            }
        }
        Ok(res)
    }

    fn query_cooperative_matrix_limits(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<Vec<CoopMatrixShape>> {
        unsafe {
            let fp = (*instance)
                .commands()
                .get_physical_device_cooperative_matrix_properties_khr;

            let mut count: u32 = 0;
            let result = fp(physical_device, &mut count, std::ptr::null_mut());
            if result != vk::Result::SUCCESS || count == 0 {
                return None;
            }

            let mut properties =
                Vec::<vk::CooperativeMatrixPropertiesKHR>::with_capacity(count as usize);
            let result = fp(physical_device, &mut count, properties.as_mut_ptr());
            if result != vk::Result::SUCCESS {
                return None;
            }
            properties.set_len(count as usize);

            let shapes: Vec<CoopMatrixShape> = properties
                .iter()
                .map(|p| CoopMatrixShape {
                    m: p.m_size,
                    n: p.n_size,
                    k: p.k_size,
                    a_type: vk_to_onnx_dtype(p.a_type).unwrap_or(DataType::Undefined),
                    b_type: vk_to_onnx_dtype(p.b_type).unwrap_or(DataType::Undefined),
                    c_type: vk_to_onnx_dtype(p.c_type).unwrap_or(DataType::Undefined),
                    result_type: vk_to_onnx_dtype(p.result_type).unwrap_or(DataType::Undefined),
                })
                .collect();

            Some(shapes)
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
        if self.shader_bfloat16 {
            v.push(CString::new(Self::VK_KHR_SHADER_BFLOAT16).unwrap());
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

        if self.shader_float_16_int8 {
            let mut feat = Box::new(vk::PhysicalDeviceShaderFloat16Int8Features {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                next: ptr::null_mut(),
                shader_float16: vk::TRUE,
                shader_int8: vk::FALSE,
            });
            feat.next = head as *mut _;
            head = (&*feat) as *const _ as *mut c_void;
            holders.push(feat);
        }

        if self.storage_16bit {
            let mut feat = Box::new(vk::PhysicalDevice16BitStorageFeatures {
                s_type: vk::StructureType::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                next: ptr::null_mut(),
                storage_buffer_16bit_access: vk::TRUE,
                uniform_and_storage_buffer_16bit_access: vk::TRUE,
                storage_push_constant16: vk::FALSE,
                storage_input_output16: vk::FALSE,
            });
            feat.next = head as *mut _;
            head = (&*feat) as *const _ as *mut c_void;
            holders.push(feat);
        }

        if self.shader_bfloat16 {
            let mut feat = Box::new(vk::PhysicalDeviceShaderBfloat16FeaturesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                next: ptr::null_mut(),
                shader_b_float16_type: vk::TRUE,
                shader_b_float_16dot_product: vk::FALSE,
                shader_b_float16_cooperative_matrix: vk::TRUE,
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

    pub fn coop_matrix_shapes(&self) -> Option<Vec<CoopMatrixShape>> {
        self.cooperative_matrix.clone()
    }

    pub fn supports_fp16(&self) -> bool {
        self.shader_float_16_int8 && self.storage_16bit
    }

    pub fn supports_bf16(&self) -> bool {
        self.shader_bfloat16 && self.shader_float_16_int8
    }

    pub fn has_memory_budget(&self) -> bool {
        self.memory_budget
    }

    /// Query cooperative matrix shapes matching the given data types
    /// Returns None if cooperative matrix extension is not available
    /// Returns Some(Vec) with matching shapes (may be empty if no match)
    pub fn get_coop_matrix_sizes(
        &self,
        a_type: DataType,
        b_type: DataType,
        c_type: DataType,
        result_type: DataType,
    ) -> Option<Vec<CoopMatrixShape>> {
        self.cooperative_matrix.as_ref().map(|shapes| {
            shapes
                .iter()
                .filter(|shape| {
                    shape.a_type == a_type
                        && shape.b_type == b_type
                        && shape.c_type == c_type
                        && shape.result_type == result_type
                })
                .cloned()
                .collect()
        })
    }
}
