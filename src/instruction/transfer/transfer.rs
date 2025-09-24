use crate::{
    ComputeManager, compute::compute_manager::DeviceLocation, gpu::vk_gpu::GPU,
    instruction::instruction::Instruction, tensor_graph::tensor_graph::TensorId,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_cpu(cm);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let data = cm.tensor_read(self.src).read();
        cm.tensor_write(self.dst).write(&data);
    }
}
