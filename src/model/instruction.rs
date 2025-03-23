use std::collections::{HashMap, HashSet};

use crate::{
    compute::compute_manager::DeviceLocation,
    dataloader::error::VKMLEngineError,
    layer::execution::LayerExecution,
    tensor::{compute_tensor::ComputeTensor, tensor_desc::TensorDesc},
    tensor_graph::tensor_graph::TensorId,
};

use super::layer_connection::{LayerConnection, LayerId};

#[derive(Clone, Debug)]
pub enum Instruction {
    // Basic operations
    MatMul {
        src1: String,
        src2: String,
        dst: String,
    },
    Add {
        src1: String,
        src2: String,
        dst: String,
    },
    Sub {
        src1: String,
        src2: String,
        dst: String,
    },
    Mul {
        src1: String,
        src2: String,
        dst: String,
    },
    Div {
        src1: String,
        src2: String,
        dst: String,
    },
    Min {
        src1: String,
        src2: String,
        dst: String,
    },
    Max {
        src1: String,
        src2: String,
        dst: String,
    },

    // Convolution
    Conv2D {
        src: String,
        weights: String,
        bias: Option<String>,
        dst: String,
        stride: (usize, usize),
        padding: (usize, usize),
    },

    // Activation functions
    ReLU {
        src: String,
        dst: String,
    },
    LeakyReLU {
        src: String,
        dst: String,
        alpha: f32,
    },
    Sigmoid {
        src: String,
        dst: String,
    },
    Softmax {
        src: String,
        dst: String,
        dim: usize,
    },
    Tanh {
        src: String,
        dst: String,
    },
    GELU {
        src: String,
        dst: String,
    },
    SiLU {
        src: String,
        dst: String,
    },

    // Used to copy tensors inside a layer
    Copy {
        src: String,
        dst: String,
    },

    // Data movement
    ReadInput {
        layer_idx: usize,        // Which input port of the current layer
        layer_tensor_idx: usize, // Which output port of the source layer
        dst: String,             // Destination tensor name
    },
    TransferToDevice {
        src: String,
        dst: String,
        source_device: DeviceLocation,
        target_device: DeviceLocation,
    },

    // Data shaping
    Reshape {
        src: String,
        dst: String,
        new_shape: TensorDesc,
    },
    Concat {
        sources: Vec<String>,
        dst: String,
        dim: usize, // Dimension along which to concatenate
    },
}

impl Instruction {
    // Get all input tensor names used by this instruction
    pub fn get_input_tensor_names(&self) -> Vec<String> {
        match self {
            // Binary operations
            Self::MatMul { src1, src2, .. }
            | Self::Add { src1, src2, .. }
            | Self::Sub { src1, src2, .. }
            | Self::Mul { src1, src2, .. }
            | Self::Div { src1, src2, .. }
            | Self::Min { src1, src2, .. }
            | Self::Max { src1, src2, .. } => vec![src1.clone(), src2.clone()],

            // Conv2D with optional bias
            Self::Conv2D {
                src, weights, bias, ..
            } => {
                let mut inputs = vec![src.clone(), weights.clone()];
                if let Some(b) = bias {
                    inputs.push(b.clone());
                }
                inputs
            }

            // Unary operations
            Self::ReLU { src, .. }
            | Self::LeakyReLU { src, .. }
            | Self::Sigmoid { src, .. }
            | Self::Softmax { src, .. }
            | Self::Tanh { src, .. }
            | Self::GELU { src, .. }
            | Self::SiLU { src, .. }
            | Self::Reshape { src, .. }
            | Self::Copy { src, .. }
            | Self::TransferToDevice { src, .. } => vec![src.clone()],

            // Special cases
            Self::ReadInput { .. } => vec![],

            // Multi-input operation
            Self::Concat { sources, .. } => sources.clone(),
        }
    }

    // Get all output tensor names for this instruction
    // Currently, all instructions have a single output, but this supports future extension
    pub fn get_output_tensor_names(&self) -> Vec<String> {
        match self {
            Self::MatMul { dst, .. }
            | Self::Add { dst, .. }
            | Self::Sub { dst, .. }
            | Self::Mul { dst, .. }
            | Self::Div { dst, .. }
            | Self::Min { dst, .. }
            | Self::Max { dst, .. }
            | Self::Conv2D { dst, .. }
            | Self::ReLU { dst, .. }
            | Self::LeakyReLU { dst, .. }
            | Self::Sigmoid { dst, .. }
            | Self::Softmax { dst, .. }
            | Self::Tanh { dst, .. }
            | Self::GELU { dst, .. }
            | Self::SiLU { dst, .. }
            | Self::Reshape { dst, .. }
            | Self::ReadInput { dst, .. }
            | Self::Copy { dst, .. }
            | Self::Concat { dst, .. }
            | Self::TransferToDevice { dst, .. } => vec![dst.clone()],
            // For future multi-output instructions
            // Self::SomeMultiOutputOp { dst1, dst2, ... } => vec![dst1.clone(), dst2.clone(), ...],
        }
    }

    // Convert tensor names to TensorIds for this layer
    pub fn get_input_tensor_ids(&self, layer_id: LayerId) -> HashSet<TensorId> {
        self.get_input_tensor_names()
            .into_iter()
            .map(|name| TensorId(layer_id, name))
            .collect()
    }

    // Get all output tensor IDs for this layer
    pub fn get_output_tensor_ids(&self, layer_id: LayerId) -> HashSet<TensorId> {
        self.get_output_tensor_names()
            .into_iter()
            .map(|name| TensorId(layer_id, name))
            .collect()
    }

    // For handling ReadInput and CopyInput instructions specifically
    pub fn process_cross_layer_inputs(
        &self,
        layer_id: LayerId,
        layer_inputs: &[LayerConnection],
        layer_executions: &HashMap<LayerId, LayerExecution>,
        tensors: &HashMap<TensorId, ComputeTensor>,
    ) -> Result<HashSet<TensorId>, VKMLEngineError> {
        let mut result = HashSet::new();

        match self {
            Self::ReadInput {
                layer_idx,
                layer_tensor_idx,
                ..
            } => {
                if *layer_idx < layer_inputs.len() {
                    let conn = &layer_inputs[*layer_idx];
                    let source_layer_id = conn.get_layerid();

                    if let Some(source_exec) = layer_executions.get(&source_layer_id) {
                        if *layer_tensor_idx < source_exec.outputs.len() {
                            let output_name = &source_exec.outputs[*layer_tensor_idx];
                            let source_id = TensorId(source_layer_id, output_name.clone());

                            if tensors.contains_key(&source_id) {
                                result.insert(source_id);
                            } else {
                                return Err(VKMLEngineError::VulkanLoadError(format!(
                                    "Tensor {} not found for layer {}",
                                    output_name, source_layer_id
                                )));
                            }
                        } else {
                            return Err(VKMLEngineError::VulkanLoadError(format!(
                                "Invalid tensor index {} for layer {}",
                                layer_tensor_idx, source_layer_id
                            )));
                        }
                    } else {
                        return Err(VKMLEngineError::VulkanLoadError(format!(
                            "Layer execution not found for layer {}",
                            source_layer_id
                        )));
                    }
                }
            }
            _ => {}
        }

        Ok(result)
    }

    pub fn get_all_input_tensor_ids(
        &self,
        layer_id: LayerId,
        layer_inputs: &[LayerConnection],
        layer_executions: &HashMap<LayerId, LayerExecution>,
        tensors: &HashMap<TensorId, ComputeTensor>,
    ) -> Result<HashSet<TensorId>, VKMLEngineError> {
        // Determine how to handle inputs based on instruction type
        match self {
            // For ReadInput and CopyInput, handle cross-layer inputs
            Self::ReadInput {
                layer_idx,
                layer_tensor_idx,
                ..
            } => {
                let mut inputs = HashSet::new();

                if *layer_idx < layer_inputs.len() {
                    let conn = &layer_inputs[*layer_idx];
                    let source_layer_id = conn.get_layerid();

                    if let Some(source_exec) = layer_executions.get(&source_layer_id) {
                        if *layer_tensor_idx < source_exec.outputs.len() {
                            let output_name = &source_exec.outputs[*layer_tensor_idx];
                            let source_id = TensorId(source_layer_id, output_name.clone());

                            if tensors.contains_key(&source_id) {
                                inputs.insert(source_id);
                            } else {
                                return Err(VKMLEngineError::VulkanLoadError(format!(
                                    "Tensor {} not found for layer {}",
                                    output_name, source_layer_id
                                )));
                            }
                        } else {
                            return Err(VKMLEngineError::VulkanLoadError(format!(
                                "Invalid tensor index {} for layer {}",
                                layer_tensor_idx, source_layer_id
                            )));
                        }
                    } else {
                        return Err(VKMLEngineError::VulkanLoadError(format!(
                            "Layer execution not found for layer {}",
                            source_layer_id
                        )));
                    }
                }

                Ok(inputs)
            }

            // For all other instructions, use local input tensors
            _ => Ok(self.get_input_tensor_ids(layer_id)),
        }
    }
}
