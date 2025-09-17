use crate::{
    dataloader::error::VKMLError,
    instruction::{self, conv::conv::AutoPad, instruction::Instruction},
    tensor::{desc::TensorDesc, tensor::Tensor},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use onnx_extractor::{
    AttributeValue, OnnxModel, OperationInfo as OnnxOperationInfo, TensorInfo as OnnxTensorInfo,
};
use std::{collections::HashMap, sync::RwLock};

pub struct OnnxParser;

impl OnnxParser {
    /// Convert ONNX model to TensorGraph
    pub fn parse_onnx_model(onnx_model: &OnnxModel) -> Result<TensorGraph, VKMLError> {
        let mut tensors = Vec::new();
        let mut operations: Vec<Box<dyn Instruction>> = Vec::new();
        let mut tensor_name_to_id: HashMap<String, TensorId> = HashMap::new();

        let mut memory_requirements = 0;

        // Create tensors from ONNX model
        for (name, onnx_tensor) in &onnx_model.tensors {
            let tensor_desc = Self::convert_onnx_tensor_to_desc(onnx_tensor)?;
            memory_requirements += tensor_desc.size_in_bytes();
            let compute_tensor = if onnx_tensor.has_data() {
                Tensor::new_cpu(tensor_desc, onnx_tensor.get_raw_data().unwrap())
            } else {
                Tensor::new_unallocated(tensor_desc)
            };

            let tensor_id = tensors.len();
            tensors.push(RwLock::new(compute_tensor));
            tensor_name_to_id.insert(name.clone(), tensor_id);
        }

        // Create operations from ONNX nodes; fail fast if an op isn't supported
        for onnx_op in &onnx_model.operations {
            let instruction =
                Self::convert_onnx_operation_to_instruction(onnx_op, &tensor_name_to_id)?;
            operations.push(instruction);
        }

        // Map input/output tensor names to IDs
        let input_tensors: Vec<TensorId> = onnx_model
            .inputs
            .iter()
            .filter_map(|name| tensor_name_to_id.get(name).copied())
            .collect();

        let output_tensors: Vec<TensorId> = onnx_model
            .outputs
            .iter()
            .filter_map(|name| tensor_name_to_id.get(name).copied())
            .collect();

        let tensor_to_layer = vec![None; tensors.len()];
        let operation_to_layer = vec![0; operations.len()];

        Ok(TensorGraph {
            tensors,
            operations,
            input_tensors,
            output_tensors,
            tensor_to_layer,
            operation_to_layer,
            memory_requirements,
        })
    }

    fn convert_onnx_tensor_to_desc(onnx_tensor: &OnnxTensorInfo) -> Result<TensorDesc, VKMLError> {
        // Convert i64 dimensions to usize, handling dynamic dimensions
        let dims: Result<Vec<i64>, VKMLError> = onnx_tensor.shape
        .iter()
        .map(|&dim| {
            if dim <= 0 {
                Err(VKMLError::OnnxImporterError(format!(
                    "Zero/Dynamic dimension {} in tensor '{}' is not supported. All dimensions must be concrete positive values.",
                    dim, onnx_tensor.name
                )))
            } else {
                Ok(dim)
            }
        })
        .collect();

        let dims = dims?;

        if dims.is_empty() {
            return Err(VKMLError::OnnxImporterError(format!(
                "Tensor '{}' has empty dimensions",
                onnx_tensor.name
            )));
        }

        Ok(TensorDesc::new_with_type(dims, onnx_tensor.data_type))
    }

    fn convert_onnx_operation_to_instruction(
        onnx_op: &OnnxOperationInfo,
        tensor_map: &HashMap<String, TensorId>,
    ) -> Result<Box<dyn Instruction>, VKMLError> {
        // Resolve tensor names to IDs
        let input_ids: Result<Vec<TensorId>, VKMLError> = onnx_op
            .inputs
            .iter()
            .map(|name| {
                tensor_map.get(name).copied().ok_or_else(|| {
                    VKMLError::OnnxImporterError(format!(
                        "Input tensor '{}' not found for operation '{}'",
                        name, onnx_op.name
                    ))
                })
            })
            .collect();
        let input_ids = input_ids?;

        let output_ids: Result<Vec<TensorId>, VKMLError> = onnx_op
            .outputs
            .iter()
            .map(|name| {
                tensor_map.get(name).copied().ok_or_else(|| {
                    VKMLError::OnnxImporterError(format!(
                        "Output tensor '{}' not found for operation '{}'",
                        name, onnx_op.name
                    ))
                })
            })
            .collect();
        let output_ids = output_ids?;

        Self::create_instruction_from_onnx_op(
            &onnx_op.op_type,
            input_ids,
            output_ids,
            &onnx_op.attributes,
        )
    }

    fn create_instruction_from_onnx_op(
        op_type: &str,
        input_ids: Vec<TensorId>,
        output_ids: Vec<TensorId>,
        attributes: &HashMap<String, AttributeValue>,
    ) -> Result<Box<dyn Instruction>, VKMLError> {
        match op_type {
            "MatMul" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "MatMul requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::matmul(
                    input_ids[0],
                    input_ids[1],
                    output_ids[0],
                ))
            }
            "Add" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Add requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::add(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Sub" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Sub requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::sub(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Mul" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Mul requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::mul(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Div" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Div requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::div(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Max" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Max currently supports exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::max(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Min" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Min currently supports exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::min(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Relu" => {
                if input_ids.len() != 1 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Relu requires exactly 1 input and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(instruction::relu(input_ids[0], output_ids[0]))
            }
            "Conv" => {
                // Expect: input, weights, optional bias -> output
                if input_ids.len() < 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Conv requires at least 2 inputs (input, weights) and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }

                let src = input_ids[0];
                let weights = input_ids[1];
                let bias = input_ids.get(2).copied();
                let dst = output_ids[0];

                // Simplified parsing: map ONNX attributes directly into instruction fields.
                let mut strides: Vec<usize> = Vec::new();
                let mut dilations: Vec<usize> = Vec::new();
                let mut kernel_shape: Vec<usize> = Vec::new();
                let mut pads: Vec<usize> = Vec::new();
                let mut groups: usize = 1;

                // Helper: extract ints from AttributeValue as Vec<i64>
                let attr_to_vec = |a: &AttributeValue| -> Option<Vec<i64>> {
                    match a {
                        AttributeValue::Ints(v) => Some(v.clone()),
                        AttributeValue::Int(i) => Some(vec![*i]),
                        _ => None,
                    }
                };

                if let Some(val) = attributes.get("strides") {
                    if let Some(v) = attr_to_vec(val) {
                        strides = v.iter().map(|x| *x as usize).collect();
                    }
                }

                if let Some(val) = attributes.get("dilations") {
                    if let Some(v) = attr_to_vec(val) {
                        dilations = v.iter().map(|x| *x as usize).collect();
                    }
                }

                if let Some(val) = attributes.get("kernel_shape") {
                    if let Some(v) = attr_to_vec(val) {
                        kernel_shape = v.iter().map(|x| *x as usize).collect();
                    }
                }

                // Parse auto_pad per ONNX (default NOTSET)
                let mut auto_pad: Option<AutoPad> = None;
                if let Some(val) = attributes.get("auto_pad") {
                    if let AttributeValue::String(s) = val {
                        auto_pad = match s.as_str() {
                            "VALID" => Some(AutoPad::Valid),
                            "SAME_UPPER" => Some(AutoPad::SameUpper),
                            "SAME_LOWER" => Some(AutoPad::SameLower),
                            "NOTSET" | "" => Some(AutoPad::NotSet),
                            _ => None,
                        };
                    }
                }
                let auto_pad_val = auto_pad.unwrap_or(AutoPad::NotSet);

                // pads: only allowed when auto_pad == NOTSET
                if let Some(val) = attributes.get("pads") {
                    if auto_pad_val != AutoPad::NotSet {
                        return Err(VKMLError::OnnxImporterError(
                            "Conv: 'pads' and 'auto_pad' cannot be used together".to_string(),
                        ));
                    }
                    if let Some(pv) = attr_to_vec(val) {
                        if pv.iter().any(|x| *x < 0) {
                            return Err(VKMLError::OnnxImporterError(
                                "Pads must be non-negative for Conv operation".to_string(),
                            ));
                        }
                        if pv.len() % 2 != 0 {
                            return Err(VKMLError::OnnxImporterError(
                                "Invalid 'pads' attribute length for Conv operation".to_string(),
                            ));
                        }
                        pads = pv.iter().map(|x| *x as usize).collect();
                    }
                }

                if let Some(val) = attributes.get("group") {
                    if let AttributeValue::Int(g) = val {
                        groups = *g as usize;
                    }
                }

                Ok(instruction::conv(
                    src,
                    weights,
                    bias,
                    dst,
                    auto_pad_val,
                    dilations,
                    groups,
                    kernel_shape,
                    pads,
                    strides,
                ))
            }
            unsupported => Err(VKMLError::OnnxImporterError(format!(
                "Operation '{}' is not implemented",
                unsupported
            ))),
        }
    }
}
