use crate::{
    compute::compute_manager::DeviceLocation,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

use super::instruction::Instruction;

#[derive(Clone)]
pub struct TransferToDeviceInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub source_device: DeviceLocation,
    pub target_device: DeviceLocation,
}

impl Debug for TransferToDeviceInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "TransferToDevice(src={}, dst={}, from={:?}, to={:?})",
            self.src, self.dst, self.source_device, self.target_device
        )
    }
}

impl Instruction for TransferToDeviceInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn create_command_buffer(
        &self,
        _gpu: &GPU,
        _command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get the tensors from the tensor graph
        let src_tensor = tensor_graph.tensors.get(self.src).unwrap();
        let dst_tensor = tensor_graph.tensors.get(self.dst).unwrap();

        // Get the raw bytes from the source tensor and write them to destination
        let data = src_tensor.data.get_data();
        dst_tensor.data.update_data(data);

        // No need for a command buffer for data transfer
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) {
        let data = tensor_graph.tensors[self.src].data.get_data();
        tensor_graph.tensors[self.dst].data.update_data(data);
    }
}
