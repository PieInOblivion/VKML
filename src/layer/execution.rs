use std::collections::HashMap;

use crate::{model::instruction::Instruction, tensor::tensor_desc::TensorDesc};

pub struct LayerExecution {
    pub tensors: HashMap<String, TensorDesc>,
    pub instructions: Vec<Instruction>,
    pub outputs: Vec<String>
}