use onnx_extractor::DataType;
use vk::ComponentTypeKHR as Ct;
use vulkanalia::vk;

/// Returns None for unknown / unsupported mappings
pub fn vk_to_onnx_dtype(t: vk::ComponentTypeKHR) -> Option<DataType> {
    match t {
        Ct::SINT8 => Some(DataType::Int8),
        Ct::UINT8 => Some(DataType::Uint8),
        Ct::SINT16 => Some(DataType::Int16),
        Ct::UINT16 => Some(DataType::Uint16),
        Ct::SINT32 => Some(DataType::Int32),
        Ct::UINT32 => Some(DataType::Uint32),
        Ct::SINT64 => Some(DataType::Int64),
        Ct::UINT64 => Some(DataType::Uint64),
        Ct::FLOAT16 => Some(DataType::Float16),
        Ct::FLOAT32 => Some(DataType::Float),
        Ct::FLOAT64 => Some(DataType::Double),
        Ct::BFLOAT16 => Some(DataType::Bfloat16),
        Ct::FLOAT8_E4M3_EXT => Some(DataType::Float8e4m3fn),
        Ct::FLOAT8_E5M2_EXT => Some(DataType::Float8e5m2),
        _ => None,
    }
}

pub fn vk_bool32_to_bool(value: vk::Bool32) -> bool {
    value == vk::TRUE
}

pub fn bool_to_vk_bool32(value: bool) -> vk::Bool32 {
    if value { vk::TRUE } else { vk::FALSE }
}
