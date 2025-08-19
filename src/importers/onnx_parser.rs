use crate::{
    dataloader::{data_type::DataType as VkmlDataType, error::VKMLError},
    instruction::{factory::Instructions, instruction::Instruction},
    tensor::{compute_tensor::ComputeTensor, tensor_desc::TensorDesc},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use onnx_extractor::{
    AttributeValue, OnnxModel, OperationInfo as OnnxOperationInfo, TensorInfo as OnnxTensorInfo,
};
use std::collections::HashMap;

pub struct OnnxParser;

impl OnnxParser {
    /// Convert ONNX model to TensorGraph
    pub fn parse_onnx_model(onnx_model: &OnnxModel) -> Result<TensorGraph, VKMLError> {
        let mut tensors = Vec::new();
        let mut operations: Vec<Box<dyn Instruction>> = Vec::new();
        let mut tensor_name_to_id: HashMap<String, TensorId> = HashMap::new();

        // Create tensors from ONNX model
        for (name, onnx_tensor) in &onnx_model.tensors {
            let tensor_desc = Self::convert_onnx_tensor_to_desc(onnx_tensor)?;
            let compute_tensor = if onnx_tensor.has_data() {
                let data = Self::extract_tensor_data(onnx_tensor)?;
                ComputeTensor::new_cpu(tensor_desc, data)
            } else {
                ComputeTensor::new_unallocated(tensor_desc)
            };

            let tensor_id = tensors.len();
            tensors.push(compute_tensor);
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
        })
    }

    fn convert_onnx_tensor_to_desc(onnx_tensor: &OnnxTensorInfo) -> Result<TensorDesc, VKMLError> {
        // Currently only Float32 tensors are supported end-to-end
        let data_type = match &onnx_tensor.data_type {
            onnx_extractor::DataType::Float32 => VkmlDataType::F32,
            unsupported => {
                return Err(VKMLError::OnnxImporterError(format!(
                    "ONNX data type {:?} is not supported (expected Float32)",
                    unsupported
                )));
            }
        };

        // Convert i64 dimensions to usize, handling dynamic dimensions
        let dims: Result<Vec<usize>, VKMLError> = onnx_tensor.shape
        .iter()
        .map(|&dim| {
            if dim < 0 {
                Err(VKMLError::OnnxImporterError(format!(
                    "Dynamic dimension {} in tensor '{}' is not supported. All dimensions must be concrete positive values.",
                    dim, onnx_tensor.name
                )))
            } else if dim == 0 {
                Err(VKMLError::OnnxImporterError(format!(
                    "Zero dimension in tensor '{}' is not supported.", 
                    onnx_tensor.name
                )))
            } else {
                // Safe conversion from positive i64 to usize
                Ok(dim as usize)
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

        Ok(TensorDesc::new_with_type(dims, data_type))
    }

    fn extract_tensor_data(onnx_tensor: &OnnxTensorInfo) -> Result<Vec<f32>, VKMLError> {
        match &onnx_tensor.data_type {
            onnx_extractor::DataType::Float32 => onnx_tensor.get_f32_data().map_err(|e| {
                VKMLError::OnnxImporterError(format!(
                    "Failed to extract f32 data from tensor '{}': {}",
                    onnx_tensor.name, e
                ))
            }),
            unsupported => Err(VKMLError::OnnxImporterError(format!(
                "Tensor data extraction for {:?} not implemented",
                unsupported
            ))),
        }
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
        _attributes: &HashMap<String, AttributeValue>,
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
                Ok(Instructions::matmul(
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
                Ok(Instructions::add(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Sub" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Sub requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::sub(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Mul" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Mul requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::mul(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Div" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Div requires exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::div(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Max" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Max currently supports exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::max(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Min" => {
                if input_ids.len() != 2 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Min currently supports exactly 2 inputs and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::min(input_ids[0], input_ids[1], output_ids[0]))
            }
            "Relu" => {
                if input_ids.len() != 1 || output_ids.len() != 1 {
                    return Err(VKMLError::OnnxImporterError(format!(
                        "Relu requires exactly 1 input and 1 output, got {} inputs and {} outputs",
                        input_ids.len(),
                        output_ids.len()
                    )));
                }
                Ok(Instructions::relu(input_ids[0], output_ids[0]))
            }
            unsupported => Err(VKMLError::OnnxImporterError(format!(
                "Operation '{}' is not implemented",
                unsupported
            ))),
        }
    }
}
