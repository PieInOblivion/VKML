use ash::{Device, vk};
use std::collections::HashMap;

// Precompiled SPIR-V shader bytes
const F32_ADD_SHADER: &[u8] = include_bytes!("../shaders/f32_add.spv");
const F32_SUB_SHADER: &[u8] = include_bytes!("../shaders/f32_sub.spv");
const F32_MUL_SHADER: &[u8] = include_bytes!("../shaders/f32_mul.spv");
const F32_DIV_SHADER: &[u8] = include_bytes!("../shaders/f32_div.spv");
const F32_MAX_SHADER: &[u8] = include_bytes!("../shaders/f32_max.spv");
const F32_MIN_SHADER: &[u8] = include_bytes!("../shaders/f32_min.spv");

const F32_ADD_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_add_inplace.spv");
const F32_SUB_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_sub_inplace.spv");
const F32_MUL_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_mul_inplace.spv");
const F32_DIV_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_div_inplace.spv");
const F32_MAX_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_max_inplace.spv");
const F32_MIN_INPLACE_SHADER: &[u8] = include_bytes!("../shaders/f32_min_inplace.spv");

const F32_RELU_SHADER: &[u8] = include_bytes!("../shaders/f32_relu.spv");
const F32_LEAKY_RELU_SHADER: &[u8] = include_bytes!("../shaders/f32_leaky_relu.spv");
const F32_SIGMOID_SHADER: &[u8] = include_bytes!("../shaders/f32_sigmoid.spv");
const F32_SOFTMAX_SHADER: &[u8] = include_bytes!("../shaders/f32_softmax.spv");
const F32_TANH_SHADER: &[u8] = include_bytes!("../shaders/f32_tanh.spv");
const F32_GELU_SHADER: &[u8] = include_bytes!("../shaders/f32_gelu.spv");
const F32_SILU_SHADER: &[u8] = include_bytes!("../shaders/f32_silu.spv");

const F32_CONV2D_SHADER: &[u8] = include_bytes!("../shaders/f32_conv2d.spv");

const F32_MATMUL_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul.spv");
const F32_MATMUL_1D_2D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_1d_2d.spv");
const F32_MATMUL_2D_1D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_2d_1d.spv");
const F32_MATMUL_2D_2D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_2d_2d.spv");
const F32_MATMUL_2D_3D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_2d_3d.spv");
const F32_MATMUL_3D_2D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_3d_2d.spv");
const F32_MATMUL_3D_3D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_3d_3d.spv");
const F32_MATMUL_3D_1D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_3d_1d.spv");
const F32_MATMUL_1D_3D_SHADER: &[u8] = include_bytes!("../shaders/f32_matmul_1d_3d.spv");

const F32_INIT_XAVIER_SHADER: &[u8] = include_bytes!("../shaders/f32_init_xavier.spv");
const F32_INIT_HE_SHADER: &[u8] = include_bytes!("../shaders/f32_init_he.spv");
const F32_INIT_LECUN_SHADER: &[u8] = include_bytes!("../shaders/f32_init_lecun.spv");
const F32_INIT_UNIFORM_SHADER: &[u8] = include_bytes!("../shaders/f32_init_uniform.spv");
const F32_INIT_NORMAL_SHADER: &[u8] = include_bytes!("../shaders/f32_init_normal.spv");
const F32_INIT_CONSTANT_SHADER: &[u8] = include_bytes!("../shaders/f32_init_constant.spv");

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GPUMemoryOperation {
    Addition,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    AdditionInplace,
    SubtractInplace,
    MultiplyInplace,
    DivideInplace,
    MaximumInplace,
    MinimumInplace,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softmax,
    Tanh,
    GELU,
    SiLU,
    Conv2D,
    MatMul,
    MatMul1D2D,
    MatMul2D1D,
    MatMul2D2D,
    MatMul2D3D,
    MatMul3D2D,
    MatMul3D3D,
    MatMul3D1D,
    MatMul1D3D,
    InitXavier,
    InitHe,
    InitLeCun,
    InitUniform,
    InitNormal,
    InitConstant,
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
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 128, // TODO: For now, have 128 bytes of push constant space
        };

        let pipeline_layout = unsafe {
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                _marker: std::marker::PhantomData,
            };

            device.create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let mut pipelines = HashMap::new();

        // Element-wise
        pipelines.insert(
            GPUMemoryOperation::Addition,
            Self::create_pipeline(device, pipeline_layout, F32_ADD_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Subtract,
            Self::create_pipeline(device, pipeline_layout, F32_SUB_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Multiply,
            Self::create_pipeline(device, pipeline_layout, F32_MUL_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Divide,
            Self::create_pipeline(device, pipeline_layout, F32_DIV_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Maximum,
            Self::create_pipeline(device, pipeline_layout, F32_MAX_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Minimum,
            Self::create_pipeline(device, pipeline_layout, F32_MIN_SHADER)?,
        );

        // Element-wise in place
        pipelines.insert(
            GPUMemoryOperation::AdditionInplace,
            Self::create_pipeline(device, pipeline_layout, F32_ADD_INPLACE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::SubtractInplace,
            Self::create_pipeline(device, pipeline_layout, F32_SUB_INPLACE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MultiplyInplace,
            Self::create_pipeline(device, pipeline_layout, F32_MUL_INPLACE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::DivideInplace,
            Self::create_pipeline(device, pipeline_layout, F32_DIV_INPLACE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MaximumInplace,
            Self::create_pipeline(device, pipeline_layout, F32_MAX_INPLACE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MinimumInplace,
            Self::create_pipeline(device, pipeline_layout, F32_MIN_INPLACE_SHADER)?,
        );

        // Activations
        pipelines.insert(
            GPUMemoryOperation::ReLU,
            Self::create_pipeline(device, pipeline_layout, F32_RELU_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::LeakyReLU,
            Self::create_pipeline(device, pipeline_layout, F32_LEAKY_RELU_SHADER)?,
        );
        // Added activation pipelines
        pipelines.insert(
            GPUMemoryOperation::Sigmoid,
            Self::create_pipeline(device, pipeline_layout, F32_SIGMOID_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Softmax,
            Self::create_pipeline(device, pipeline_layout, F32_SOFTMAX_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::Tanh,
            Self::create_pipeline(device, pipeline_layout, F32_TANH_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::GELU,
            Self::create_pipeline(device, pipeline_layout, F32_GELU_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::SiLU,
            Self::create_pipeline(device, pipeline_layout, F32_SILU_SHADER)?,
        );

        // Conv2D
        pipelines.insert(
            GPUMemoryOperation::Conv2D,
            Self::create_pipeline(device, pipeline_layout, F32_CONV2D_SHADER)?,
        );

        // Fallback generic N x N matmul
        pipelines.insert(
            GPUMemoryOperation::MatMul,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_SHADER)?,
        );

        // Specific dim MatMul
        pipelines.insert(
            GPUMemoryOperation::MatMul1D2D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_1D_2D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul2D1D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_2D_1D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul2D2D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_2D_2D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul2D3D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_2D_3D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul3D2D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_3D_2D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul3D3D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_3D_3D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul3D1D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_3D_1D_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::MatMul1D3D,
            Self::create_pipeline(device, pipeline_layout, F32_MATMUL_1D_3D_SHADER)?,
        );

        // GPU Weight Initialization Operations
        pipelines.insert(
            GPUMemoryOperation::InitXavier,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_XAVIER_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::InitHe,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_HE_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::InitLeCun,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_LECUN_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::InitUniform,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_UNIFORM_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::InitNormal,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_NORMAL_SHADER)?,
        );
        pipelines.insert(
            GPUMemoryOperation::InitConstant,
            Self::create_pipeline(device, pipeline_layout, F32_INIT_CONSTANT_SHADER)?,
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
                aligned_code = padded
                    .chunks_exact(4)
                    .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
            } else {
                aligned_code = std::slice::from_raw_parts(
                    shader_code.as_ptr() as *const u32,
                    shader_code.len() / 4,
                )
                .to_vec();
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

            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))?[0];

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
