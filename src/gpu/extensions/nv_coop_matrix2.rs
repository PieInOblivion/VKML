use crate::utils::vk_to_onnx_dtype::{vk_bool32_to_bool, vk_to_onnx_dtype};
use onnx_extractor::DataType;
use std::ptr;
use vulkanalia::vk::{InstanceV1_0, InstanceV1_1};
use vulkanalia::{Instance, vk};

#[derive(Clone, Debug)]
pub struct CoopMatrixNV2Features {
    pub workgroup_scope: bool,
    pub flexible_dimensions: bool,
    pub reductions: bool,
    pub conversions: bool,
    pub per_element_operations: bool,
    pub tensor_addressing: bool,
    pub block_loads: bool,
}

#[derive(Clone, Debug)]
pub struct CoopMatrixNV2Properties {
    pub workgroup_scope_max_workgroup_size: u32,
    pub flexible_dimensions_max_dimension: u32,
    pub workgroup_scope_reserved_shared_memory: u32,
}

#[derive(Clone, Debug)]
pub struct CoopMatrixFlexibleDimensions {
    pub m_granularity: u32,
    pub n_granularity: u32,
    pub k_granularity: u32,
    pub a_type: DataType,
    pub b_type: DataType,
    pub c_type: DataType,
    pub result_type: DataType,
    pub saturating_accumulation: bool,
    pub scope: vk::ScopeKHR,
    pub workgroup_invocations: u32,
}

#[derive(Clone, Debug)]
pub struct CoopMatrixNV2Capabilities {
    pub features: CoopMatrixNV2Features,
    pub properties: CoopMatrixNV2Properties,
    pub flexible_dimensions: Vec<CoopMatrixFlexibleDimensions>,
}

pub(super) fn query_cooperative_matrix_nv2_limits(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<CoopMatrixNV2Capabilities> {
    unsafe {
        let mut coop2_features = vk::PhysicalDeviceCooperativeMatrix2FeaturesNV {
            s_type: vk::StructureType::PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
            next: ptr::null_mut(),
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            next: &mut coop2_features as *mut _ as *mut _,
            features: Default::default(),
        };

        instance.get_physical_device_features2(physical_device, &mut features2);

        let mut coop2_properties = vk::PhysicalDeviceCooperativeMatrix2PropertiesNV {
            s_type: vk::StructureType::PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV,
            next: ptr::null_mut(),
            ..Default::default()
        };

        let mut props2 = vk::PhysicalDeviceProperties2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
            next: &mut coop2_properties as *mut _ as *mut _,
            properties: Default::default(),
        };

        instance.get_physical_device_properties2(physical_device, &mut props2);

        let fp = (*instance)
            .commands()
            .get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv;

        let mut count: u32 = 0;
        let mut result = fp(physical_device, &mut count, ptr::null_mut());
        if result != vk::Result::SUCCESS {
            return None;
        }

        let mut flexible_dimensions_raw: Vec<vk::CooperativeMatrixFlexibleDimensionsPropertiesNV> =
            Vec::with_capacity(count as usize);

        if count > 0 {
            result = fp(
                physical_device,
                &mut count,
                flexible_dimensions_raw.as_mut_ptr(),
            );
            if result != vk::Result::SUCCESS {
                return None;
            }
            flexible_dimensions_raw.set_len(count as usize);
        }

        let flexible_dimensions = flexible_dimensions_raw
            .into_iter()
            .map(|p| CoopMatrixFlexibleDimensions {
                m_granularity: p.m_granularity,
                n_granularity: p.n_granularity,
                k_granularity: p.k_granularity,
                a_type: vk_to_onnx_dtype(p.a_type).unwrap_or(DataType::Undefined),
                b_type: vk_to_onnx_dtype(p.b_type).unwrap_or(DataType::Undefined),
                c_type: vk_to_onnx_dtype(p.c_type).unwrap_or(DataType::Undefined),
                result_type: vk_to_onnx_dtype(p.result_type).unwrap_or(DataType::Undefined),
                saturating_accumulation: vk_bool32_to_bool(p.saturating_accumulation),
                scope: p.scope,
                workgroup_invocations: p.workgroup_invocations,
            })
            .collect();

        Some(CoopMatrixNV2Capabilities {
            features: CoopMatrixNV2Features {
                workgroup_scope: vk_bool32_to_bool(
                    coop2_features.cooperative_matrix_workgroup_scope,
                ),
                flexible_dimensions: vk_bool32_to_bool(
                    coop2_features.cooperative_matrix_flexible_dimensions,
                ),
                reductions: vk_bool32_to_bool(coop2_features.cooperative_matrix_reductions),
                conversions: vk_bool32_to_bool(coop2_features.cooperative_matrix_conversions),
                per_element_operations: vk_bool32_to_bool(
                    coop2_features.cooperative_matrix_per_element_operations,
                ),
                tensor_addressing: vk_bool32_to_bool(
                    coop2_features.cooperative_matrix_tensor_addressing,
                ),
                block_loads: vk_bool32_to_bool(coop2_features.cooperative_matrix_block_loads),
            },
            properties: CoopMatrixNV2Properties {
                workgroup_scope_max_workgroup_size: coop2_properties
                    .cooperative_matrix_workgroup_scope_max_workgroup_size,
                flexible_dimensions_max_dimension: coop2_properties
                    .cooperative_matrix_flexible_dimensions_max_dimension,
                workgroup_scope_reserved_shared_memory: coop2_properties
                    .cooperative_matrix_workgroup_scope_reserved_shared_memory,
            },
            flexible_dimensions,
        })
    }
}
