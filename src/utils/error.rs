use thiserror::Error;

#[derive(Error, Debug)]
pub enum VKMLError {
    #[error("Vulkan error: {0}")]
    Vulkan(String),

    #[error("Compute Manager error: {0}")]
    ComputeManager(String),

    #[error("GPU Pool error: {0}")]
    GpuPool(String),

    #[error("Graph Scheduler error: {0}")]
    GraphScheduler(String),

    #[error("Tensor Graph error: {0}")]
    TensorGraph(String),

    #[error("Graph Model error: {0}")]
    GraphModel(String),

    #[error("Layer error: {0}")]
    Layer(String),

    #[error("Instruction error: {0}")]
    Instruction(String),

    #[error("Onnx Importer error: {0}")]
    OnnxImporter(String),
}

// Convert vk::Result (Vulkan return codes) into VKMLError
impl From<vulkanalia::vk::Result> for VKMLError {
    fn from(r: vulkanalia::vk::Result) -> Self {
        VKMLError::Vulkan(format!("vk::Result: {:?}", r))
    }
}

impl From<vulkanalia::vk::ErrorCode> for VKMLError {
    fn from(c: vulkanalia::vk::ErrorCode) -> Self {
        VKMLError::Vulkan(format!("vk::ErrorCode: {:?}", c))
    }
}
