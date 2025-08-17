//! VKML - High-level abstractions for ML model development using Vulkan compute
//!
//! This library provides universal compute utilisation across different hardware vendors
//! with a focus on performance and ease of use.

mod gpu;

mod compute;

mod model;

mod dataloader;

mod layer;

mod tensor;

mod instruction;

mod tensor_graph;

mod utils;

pub use compute::compute_manager::ComputeManager;
pub use dataloader::{config::DataLoaderConfig, data_batch::DataBatch};
pub use layer::factory::Layers;
pub use model::{graph_model::GraphModel, layer_connection::LayerConnection};
