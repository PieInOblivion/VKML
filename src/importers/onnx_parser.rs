use crate::{
    instruction::{self, conv::conv::AutoPad, instruction::Instruction},
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
    utils::error::VKMLError,
};
use onnx_extractor::{AttributeValue, OnnxModel, OnnxOperation, OnnxTensor};
use std::collections::HashMap;

pub struct OnnxParser;

impl OnnxParser {
    /// Convert ONNX model to TensorGraph
    pub fn parse_onnx_model(
        onnx_model: OnnxModel,
    ) -> Result<(TensorGraph, Vec<Option<Box<[u8]>>>), VKMLError> {
        let mut tensor_descs = Vec::new();
        let mut tensor_bytes = Vec::new();
        let mut operations: Vec<Box<dyn Instruction>> = Vec::new();
        let mut tensor_name_to_id: HashMap<String, TensorId> = HashMap::new();

        let mut memory_requirements = 0;

        // Create tensors from ONNX model
        for (name, onnx_tensor) in onnx_model.tensors {
            let onnx_tensor_desc = Self::convert_onnx_tensor_to_desc(&onnx_tensor)?;
            memory_requirements += onnx_tensor_desc.size_in_bytes();

            tensor_descs.push(onnx_tensor_desc.clone());

            let tensor_opt = onnx_tensor.into_bytes().ok();

            tensor_bytes.push(tensor_opt);

            tensor_name_to_id.insert(name.clone(), tensor_descs.len() - 1);
        }

        // Create operations from ONNX nodes; fail fast if an op isn't supported
        for onnx_op in &onnx_model.operations {
            let instruction = Self::convert_onnx_operation_to_instruction(
                onnx_op,
                &tensor_name_to_id,
                &tensor_bytes,
            )?;
            operations.push(instruction);
        }

        // Map input/output tensor names to IDs
        let input_tensor_ids: Vec<TensorId> = onnx_model
            .inputs
            .iter()
            .filter_map(|name| tensor_name_to_id.get(name).copied())
            .collect();

        let output_tensor_ids: Vec<TensorId> = onnx_model
            .outputs
            .iter()
            .filter_map(|name| tensor_name_to_id.get(name).copied())
            .collect();

        let tensor_to_layer = vec![None; tensor_descs.len()];
        let operation_to_layer = vec![0; operations.len()];

        Ok((
            TensorGraph {
                tensor_descs,
                operations,
                input_tensor_ids,
                output_tensor_ids,
                tensor_to_layer,
                operation_to_layer,
                memory_requirements,
            },
            tensor_bytes,
        ))
    }

    fn convert_onnx_tensor_to_desc(onnx_tensor: &OnnxTensor) -> Result<TensorDesc, VKMLError> {
        // Convert i64 dimensions to usize, handling dynamic dimensions
        let dims = onnx_tensor.shape
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
        .collect::<Result<Vec<i64>, VKMLError>>()?;

        if dims.is_empty() {
            return Err(VKMLError::OnnxImporterError(format!(
                "Tensor '{}' has empty dimensions",
                onnx_tensor.name
            )));
        }

        Ok(TensorDesc::new(dims, onnx_tensor.data_type))
    }

    fn convert_onnx_operation_to_instruction(
        onnx_op: &OnnxOperation,
        tensor_map: &HashMap<String, TensorId>,
        tensors: &[Option<Box<[u8]>>],
    ) -> Result<Box<dyn Instruction>, VKMLError> {
        // Resolve tensor names to IDs
        let input_ids = onnx_op
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
            .collect::<Result<Vec<TensorId>, VKMLError>>()?;

        let output_ids = onnx_op
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
            .collect::<Result<Vec<TensorId>, VKMLError>>()?;

        let attributes = &onnx_op.attributes;

        // Small helpers to extract attribute values
        let attr_to_vec = |a: &AttributeValue| -> Option<Vec<i64>> {
            match a {
                AttributeValue::Ints(v) => Some(v.clone()),
                AttributeValue::Int(i) => Some(vec![*i]),
                _ => None,
            }
        };

        let attr_to_int = |a: &AttributeValue| -> Option<i64> {
            match a {
                AttributeValue::Int(i) => Some(*i),
                _ => None,
            }
        };

        let attr_to_string = |a: &AttributeValue| -> Option<String> {
            match a {
                AttributeValue::String(s) => Some(s.clone()),
                _ => None,
            }
        };

        match &*onnx_op.op_type {
            "MatMul" => Ok(instruction::matmul(
                input_ids[0],
                input_ids[1],
                output_ids[0],
            )),
            "Concat" => {
                let axis = if let Some(a) = attributes.get("axis") {
                    attr_to_int(a).ok_or_else(|| {
                        VKMLError::OnnxImporterError(
                            "Concat: 'axis' attribute must be an int".to_string(),
                        )
                    })? as usize
                } else {
                    0usize
                };

                Ok(instruction::concat(input_ids, output_ids[0], axis))
            }
            "Reshape" => {
                let shape_id = input_ids[1];
                let raw = tensors[shape_id]
                    .as_ref()
                    .expect("Reshape parameter tensor missing");

                if !raw.len().is_multiple_of(8) {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Reshape: shape initializer has invalid raw byte length {}",
                        raw.len()
                    )));
                }

                let mut shape_vec: Vec<i64> = Vec::with_capacity(raw.len() / 8);
                for chunk in raw.chunks_exact(8) {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(chunk);
                    shape_vec.push(i64::from_le_bytes(a));
                }

                let allowzero = attributes.get("allowzero").and_then(attr_to_int);
                Ok(instruction::reshape(
                    input_ids[0],
                    output_ids[0],
                    shape_vec,
                    allowzero,
                ))
            }
            "Sigmoid" => Ok(instruction::sigmoid(input_ids[0], output_ids[0])),
            "Softmax" => {
                // default axis is usually 1
                let axis = if let Some(a) = attributes.get("axis") {
                    attr_to_int(a).ok_or_else(|| {
                        VKMLError::OnnxImporterError(
                            "Softmax: 'axis' attribute must be an int".to_string(),
                        )
                    })? as usize
                } else {
                    1usize
                };
                Ok(instruction::softmax(input_ids[0], output_ids[0], axis))
            }
            "Identity" => Ok(instruction::identity(input_ids[0], output_ids[0])),
            "MaxPool" => {
                // Parse attributes similar to Conv: kernel_shape, pads, strides, dilations, auto_pad, ceil_mode
                let mut strides: Vec<usize> = Vec::new();
                let mut dilations: Vec<usize> = Vec::new();
                let mut kernel_shape: Vec<usize> = Vec::new();
                let mut pads: Vec<usize> = Vec::new();
                let mut auto_pad = AutoPad::NotSet;
                let mut ceil_mode = false;

                if let Some(val) = attributes.get("strides")
                    && let Some(v) = attr_to_vec(val)
                {
                    strides = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("dilations")
                    && let Some(v) = attr_to_vec(val)
                {
                    dilations = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("kernel_shape")
                    && let Some(v) = attr_to_vec(val)
                {
                    kernel_shape = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("pads")
                    && let Some(v) = attr_to_vec(val)
                {
                    pads = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("auto_pad")
                    && let Some(s) = attr_to_string(val)
                {
                    auto_pad = match s.as_str() {
                        "VALID" => AutoPad::Valid,
                        "SAME_UPPER" => AutoPad::SameUpper,
                        "SAME_LOWER" => AutoPad::SameLower,
                        _ => AutoPad::NotSet,
                    };
                }

                if let Some(val) = attributes.get("ceil_mode")
                    && let Some(i) = attr_to_int(val)
                {
                    ceil_mode = i != 0;
                }

                Ok(instruction::maxpool(
                    input_ids[0],
                    output_ids[0],
                    auto_pad,
                    dilations,
                    kernel_shape,
                    pads,
                    strides,
                    ceil_mode,
                ))
            }
            "Add" => Ok(instruction::add(input_ids[0], input_ids[1], output_ids[0])),
            "Sub" => Ok(instruction::sub(input_ids[0], input_ids[1], output_ids[0])),
            "Mul" => Ok(instruction::mul(input_ids[0], input_ids[1], output_ids[0])),
            "Div" => Ok(instruction::div(input_ids[0], input_ids[1], output_ids[0])),
            "Max" => Ok(instruction::max(input_ids[0], input_ids[1], output_ids[0])),
            "Min" => Ok(instruction::min(input_ids[0], input_ids[1], output_ids[0])),
            "Relu" => Ok(instruction::relu(input_ids[0], output_ids[0])),
            "Conv" => {
                // Expect: input, weights, optional bias -> output
                let src = input_ids[0];
                let weights = input_ids[1];
                let bias = input_ids.get(2).copied();
                let dst = output_ids[0];

                // Simplified parsing: map ONNX attributes directly into instruction fields.
                let mut strides: Vec<usize> = Vec::new();
                let mut dilations: Vec<usize> = Vec::new();
                let mut kernel_shape: Vec<usize> = Vec::new();
                let mut pads: Vec<usize> = Vec::new();
                let mut groups = 1;

                if let Some(val) = attributes.get("strides")
                    && let Some(v) = attr_to_vec(val)
                {
                    strides = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("dilations")
                    && let Some(v) = attr_to_vec(val)
                {
                    dilations = v.iter().map(|x| *x as usize).collect();
                }

                if let Some(val) = attributes.get("kernel_shape")
                    && let Some(v) = attr_to_vec(val)
                {
                    kernel_shape = v.iter().map(|x| *x as usize).collect();
                }

                // Parse auto_pad per ONNX (default NOTSET)
                let mut auto_pad: Option<AutoPad> = None;
                if let Some(val) = attributes.get("auto_pad")
                    && let AttributeValue::String(s) = val
                {
                    auto_pad = match s.as_str() {
                        "VALID" => Some(AutoPad::Valid),
                        "SAME_UPPER" => Some(AutoPad::SameUpper),
                        "SAME_LOWER" => Some(AutoPad::SameLower),
                        "NOTSET" | "" => Some(AutoPad::NotSet),
                        _ => None,
                    };
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

                if let Some(val) = attributes.get("group")
                    && let AttributeValue::Int(g) = val
                {
                    groups = *g;
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
