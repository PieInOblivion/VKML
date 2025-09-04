use std::collections::HashMap;

use crate::{
    instruction::instruction::Instruction, tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};

pub struct LayerExecution {
    pub tensors: Vec<TensorDesc>,                // Tensor descriptors
    pub instructions: Vec<Box<dyn Instruction>>, // Layer-local instructions
    pub outputs: Vec<TensorId>,                  // Output tensor IDs
    pub input_mappings: HashMap<TensorId, (usize, TensorId)>, // Maps local tensor to (input_connection_idx, output_idx)
}
