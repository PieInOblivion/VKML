use thiserror::Error;

// TODO: Reduce or shrink usage of VulkanLoadError

#[derive(Error, Debug)]
pub enum VKMLError {
    #[error("Vulkan error: {0}")]
    VulkanLoadError(String),

    #[error("{0}")]
    OnnxImporterError(String),

    #[error("{0}")]
    Generic(String),
}
