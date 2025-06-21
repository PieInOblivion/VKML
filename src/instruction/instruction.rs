use crate::{
    dataloader::error::VKMLEngineError,
    execution::execution_mode::ExecutionMode,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::Debug;
use vulkanalia::vk;

pub trait Instruction: Debug {
    // Get all input tensor IDs used by this instruction
    fn get_input_tensor_ids(&self) -> Vec<TensorId>;

    // Get all output tensor IDs for this instruction
    fn get_output_tensor_ids(&self) -> Vec<TensorId>;

    // Remap tensor IDs (used during graph construction)
    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]);

    // Create a Vulkan command buffer for this instruction
    fn create_command_buffer(
        &self,
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err(Box::new(VKMLEngineError::VulkanLoadError(format!(
            "GPU execution not implemented for {:?}",
            self
        ))))
    }

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        Err(VKMLEngineError::VulkanLoadError(format!(
            "CPU execution not implemented for {:?}",
            self
        )))
    }

    // Get the execution modes this instruction is used in
    // Default is only inference. Other specific modes must be manually set
    // NOTE: Not sure if this default behaviour is easiest?
    fn execution_modes(&self) -> Vec<ExecutionMode> {
        vec![ExecutionMode::Inference]
    }

    // Check if this instruction should execute for a given mode
    fn should_execute_for(&self, mode: &ExecutionMode) -> bool {
        self.execution_modes().contains(mode)
    }

    // Clone the instruction (since trait objects can't use derive(Clone))
    // Requires manual implementation for each layer
    fn clone_box(&self) -> Box<dyn Instruction>;
}

// Enable cloning for boxed instructions
impl Clone for Box<dyn Instruction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
