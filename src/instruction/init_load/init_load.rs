use crate::{
    ComputeManager, gpu::vk_gpu::GPU, instruction::instruction::Instruction,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct InitLoadInstruction {
    pub dst: TensorId,
    pub data: Vec<u8>,
    pub datatype: DataType,
}

impl Debug for InitLoadInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "InitLoad(dst={}, data_len={})",
            self.dst,
            self.data.len()
        )
    }
}

impl Instruction for InitLoadInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![]
    }
    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }
    fn remap_tensor_ids(&mut self, _new_inputs: &[TensorId], new_outputs: &[TensorId]) {
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
        let dst = cm.tensor_write(self.dst);
        match self.datatype {
            onnx_extractor::DataType::Float => {
                let out = dst.get_cpu_memory_mut_slice_or_panic();
                let copy_len = out.len().min(self.data.len());
                out[..copy_len].copy_from_slice(&self.data[..copy_len]);
            }
            _ => unimplemented!("InitLoad CPU for other types"),
        }
    }
}
