//! VKML - High-level abstractions for ML model development using Vulkan compute
//!
//! This library provides universal compute utilisation across different hardware vendors
//! with a focus on performance and ease of use.

mod utils;

mod gpu;

mod compute;
mod scheduler;

mod model;

mod layer;

mod tensor;

mod instruction;

mod tensor_graph;

mod importers;

pub use compute::compute_manager::ComputeManager;
pub use importers::onnx_parser;
pub use layer::factory::Layers;
pub use model::{graph_model::GraphModel, layer_connection::LayerConnection};
pub use tensor::Tensor;
pub use tensor::TensorDesc;
pub use utils::error;
