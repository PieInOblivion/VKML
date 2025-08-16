use thiserror::Error;

// TODO: Seperate concerns of error types
#[derive(Error, Debug)]
pub enum VKMLError {
    // Vulkan, compute and other errors
    #[error("Vulkan error: {0}")]
    VulkanLoadError(String),

    // Generic error for everything else
    #[error("{0}")]
    Generic(String),
}
