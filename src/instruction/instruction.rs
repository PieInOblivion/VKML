use crate::{ComputeManager, gpu::vk_gpu::Gpu, tensor_graph::TensorId, utils::error::VKMLError};
use std::fmt::Debug;
use vulkanalia::vk;

pub trait Instruction: Debug {
    // Get all input tensor IDs used by this instruction
    fn get_input_tensor_ids(&self) -> Vec<TensorId>;

    // Get all output tensor IDs for this instruction
    fn get_output_tensor_ids(&self) -> Vec<TensorId>;

    // Remap tensor IDs (used during graph construction)
    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]);

    // Record this instruction into an already begun command buffer
    fn record_into_command_buffer(
        &self,
        _gpu: &Gpu,
        _command_buffer: vk::CommandBuffer,
        _cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        Err(VKMLError::Vulkan(format!(
            "GPU execution not implemented for {:?}",
            self
        )))
    }

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _cm: &ComputeManager) {
        panic!("CPU execution not implemented for {:?}", self)
    }

    // Return true if this instruction must be executed on the CPU (eg transfers)
    fn must_execute_on_cpu(&self) -> bool {
        false
    }
}
