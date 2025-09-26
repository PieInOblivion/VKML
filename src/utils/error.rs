use thiserror::Error;

// TODO: Reduce or shrink usage of VulkanError

#[derive(Error, Debug)]
pub enum VKMLError {
    #[error("Vulkan error: {0}")]
    VulkanError(String),

    #[error("{0}")]
    OnnxImporterError(String),

    #[error("{0}")]
    Generic(String),
}

// Convert vk::Result (Vulkan return codes) into VKMLError
impl From<vulkanalia::vk::Result> for VKMLError {
    fn from(r: vulkanalia::vk::Result) -> Self {
        VKMLError::VulkanError(format!("vk::Result: {:?}", r))
    }
}

impl From<vulkanalia::vk::ErrorCode> for VKMLError {
    fn from(c: vulkanalia::vk::ErrorCode) -> Self {
        VKMLError::VulkanError(format!("vk::ErrorCode: {:?}", c))
    }
}
