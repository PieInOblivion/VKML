use ash::{vk, Device};
use std::collections::HashMap;

// Precompiled SPIR-V shader bytes
const F32_ADD_ARRAY_SHADER: &[u8] = include_bytes!("../shaders/f32_add_array.spv");
const F32_SUB_ARRAY_SHADER: &[u8] = include_bytes!("../shaders/f32_sub_array.spv");
const F32_MUL_ARRAY_SHADER: &[u8] = include_bytes!("../shaders/f32_mul_array.spv");
const F32_DIV_ARRAY_SHADER: &[u8] = include_bytes!("../shaders/f32_div_array.spv");

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub enum GPUMemoryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

pub struct ComputePipelines {
    pipelines: HashMap<GPUMemoryOperation, vk::Pipeline>,
    pipeline_layout: vk::PipelineLayout,
}

impl ComputePipelines {
    pub fn new(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline_layout = unsafe {
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 0,
                p_push_constant_ranges: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };

            device.create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let mut pipelines = HashMap::new();
        
        pipelines.insert(
            GPUMemoryOperation::Add,
            Self::create_pipeline(device, pipeline_layout, F32_ADD_ARRAY_SHADER)?
        );
        pipelines.insert(
            GPUMemoryOperation::Subtract,
            Self::create_pipeline(device, pipeline_layout, F32_SUB_ARRAY_SHADER)?
        );
        pipelines.insert(
            GPUMemoryOperation::Multiply,
            Self::create_pipeline(device, pipeline_layout, F32_MUL_ARRAY_SHADER)?
        );
        pipelines.insert(
            GPUMemoryOperation::Divide,
            Self::create_pipeline(device, pipeline_layout, F32_DIV_ARRAY_SHADER)?
        );

        Ok(Self {
            pipelines,
            pipeline_layout,
        })
    }

    fn create_pipeline(
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        shader_code: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let aligned_code: Vec<u32>;
            if shader_code.as_ptr().align_offset(4) != 0 {
                let mut padded = Vec::with_capacity((shader_code.len() + 3) / 4 * 4);
                padded.extend_from_slice(shader_code);
                while padded.len() % 4 != 0 {
                    padded.push(0);
                }
                aligned_code = padded.chunks_exact(4)
                    .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
            } else {
                aligned_code = std::slice::from_raw_parts(
                    shader_code.as_ptr() as *const u32,
                    shader_code.len() / 4,
                ).to_vec();
            }

            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::ShaderModuleCreateFlags::empty(),
                code_size: aligned_code.len() * 4,
                p_code: aligned_code.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let shader_module = device.create_shader_module(&shader_info, None)?;

            let entry_point = std::ffi::CString::new("main")?;
            let pipeline_info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineCreateFlags::empty(),
                stage: vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module: shader_module,
                    p_name: entry_point.as_ptr(),
                    p_specialization_info: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: -1,
                _marker: std::marker::PhantomData,
            };

            let pipeline = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ).map_err(|e| format!("Failed to create compute pipeline: {:?}", e))?[0];

            device.destroy_shader_module(shader_module, None);

            Ok(pipeline)
        }
    }

    pub fn get_pipeline(&self, op: GPUMemoryOperation) -> Option<vk::Pipeline> {
        self.pipelines.get(&op).copied()
    }

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn cleanup(&mut self, device: &Device) {
        unsafe {
            for pipeline in self.pipelines.values() {
                device.destroy_pipeline(*pipeline, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}