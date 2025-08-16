use thiserror::Error;

// TODO: Seperate concerns of error types
#[derive(Error, Debug)]
pub enum VKMLError {
    // IO and System Errors
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    // TODO: Redo error types and such
    #[error("Invalid split ratios: {message}")]
    InvalidSplitRatios { message: String },

    #[error(
        "Invalid batch index: {batch_idx}, only {max_batches} batches available for split {split_idx}"
    )]
    InvalidBatchIndex {
        batch_idx: usize,
        max_batches: usize,
        split_idx: usize,
    },

    #[error("Invalid split index: {split_idx}, only {max_splits} splits available")]
    InvalidSplitIndex { split_idx: usize, max_splits: usize },

    #[error("No images found in the dataset")]
    EmptyDataset,

    #[error("Random number generator (shuffle_seed) not set or enabled")]
    RngNotSet,

    #[error("Failed to acquire lock on RNG")]
    RngLockError,

    // Vulkan, compute and other errors
    #[error("Vulkan error: {0}")]
    VulkanLoadError(String),

    #[error("Out of memory error: {0}")]
    OutOfMemory(String),
}
