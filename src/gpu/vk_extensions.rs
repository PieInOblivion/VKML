use std::ffi::CStr;
use vulkanalia::vk;

#[derive(Clone, Debug, Default)]
pub struct VkExtensions {
    pub cooperative_matrix: bool,
    pub shader_float16_int8: bool,
    pub timeline_semaphore: bool,
    pub synchronization2: bool
}

impl VkExtensions {
    // vulkan extension strings we care about
    pub const VK_KHR_COOPERATIVE_MATRIX: &'static str = "VK_KHR_cooperative_matrix";
    pub const VK_KHR_SHADER_FLOAT16_INT8: &'static str = "VK_KHR_shader_float16_int8";
    pub const VK_KHR_TIMELINE_SEMAPHORE: &'static str = "VK_KHR_timeline_semaphore";
    pub const VK_KHR_SYNCHRONIZATION2: &'static str = "VK_KHR_synchronization2";

    // build from a slice of vk::ExtensionProperties returned by Vulkan
    pub fn from_extension_properties(props: &[vk::ExtensionProperties]) -> Self {
        let mut res = VkExtensions::default();

        for p in props {
            // conversion from the fixed-size name array
            let name_cstr = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
            let name = name_cstr.to_string_lossy();

            match name.as_ref() {
                Self::VK_KHR_COOPERATIVE_MATRIX => res.cooperative_matrix = true,
                Self::VK_KHR_SHADER_FLOAT16_INT8 => res.shader_float16_int8 = true,
                Self::VK_KHR_TIMELINE_SEMAPHORE => res.timeline_semaphore = true,
                Self::VK_KHR_SYNCHRONIZATION2 => res.synchronization2 = true,
                _ => {}
            }
        }

        res
    }
}
