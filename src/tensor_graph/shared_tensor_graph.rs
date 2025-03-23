use std::marker::{Send, Sync};

use crate::gpu::vk_gpu::GPU;

use super::tensor_graph::TensorGraph;

#[derive(Clone)]
pub struct SharedTensorGraph {
    pub tensor_graph: *const TensorGraph,
}

#[derive(Clone)]
pub struct SharedGPU {
    pub gpu: Option<*const GPU>,
}

unsafe impl Send for SharedTensorGraph {}
unsafe impl Sync for SharedTensorGraph {}

unsafe impl Send for SharedGPU {}
unsafe impl Sync for SharedGPU {}
