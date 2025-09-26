use crate::{
    ComputeManager, gpu::vk_gpu::GPU, tensor_graph::tensor_graph::TensorId, utils::error::VKMLError,
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
        _gpu: &GPU,
        _command_buffer: vk::CommandBuffer,
        _cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err(Box::new(VKMLError::VulkanError(format!(
            "GPU execution not implemented for {:?}",
            self
        ))))
    }

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _cm: &ComputeManager) {
        panic!("CPU execution not implemented for {:?}", self)
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
