macro_rules! include_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    };
}

// shader byte constants centralised here so operations can map directly to their SPIR-V
// The actual .spv files are generated into OUT_DIR by the build script and resolved
// via the include_shader! macro
const ADD_SHADER_F32: &[u8] = include_shader!("f32_add.spv");
const SUB_SHADER_F32: &[u8] = include_shader!("f32_sub.spv");
const MUL_SHADER_F32: &[u8] = include_shader!("f32_mul.spv");
const DIV_SHADER_F32: &[u8] = include_shader!("f32_div.spv");
const MAX_SHADER_F32: &[u8] = include_shader!("f32_max.spv");
const MIN_SHADER_F32: &[u8] = include_shader!("f32_min.spv");

const RELU_SHADER_F32: &[u8] = include_shader!("f32_relu.spv");
const SIGMOID_SHADER_F32: &[u8] = include_shader!("f32_sigmoid.spv");
const SOFTMAX_SHADER_F32: &[u8] = include_shader!("f32_softmax.spv");

const CONV1D_SHADER_F32: &[u8] = include_shader!("f32_conv1d.spv");
const CONV2D_SHADER_F32: &[u8] = include_shader!("f32_conv2d.spv");
const CONV3D_SHADER_F32: &[u8] = include_shader!("f32_conv3d.spv");

const MATMUL_SHADER_F32: &[u8] = include_shader!("f32_matmul.spv");
const MATMUL_1D_2D_SHADER_F32: &[u8] = include_shader!("f32_matmul_1d_2d.spv");
const MATMUL_2D_1D_SHADER_F32: &[u8] = include_shader!("f32_matmul_2d_1d.spv");
const MATMUL_2D_2D_SHADER_F32: &[u8] = include_shader!("f32_matmul_2d_2d.spv");
const MATMUL_2D_3D_SHADER_F32: &[u8] = include_shader!("f32_matmul_2d_3d.spv");
const MATMUL_3D_2D_SHADER_F32: &[u8] = include_shader!("f32_matmul_3d_2d.spv");
const MATMUL_3D_3D_SHADER_F32: &[u8] = include_shader!("f32_matmul_3d_3d.spv");
const MATMUL_3D_1D_SHADER_F32: &[u8] = include_shader!("f32_matmul_3d_1d.spv");
const MATMUL_1D_3D_SHADER_F32: &[u8] = include_shader!("f32_matmul_1d_3d.spv");

const INIT_XAVIER_SHADER_F32: &[u8] = include_shader!("f32_init_xavier.spv");
const INIT_HE_SHADER_F32: &[u8] = include_shader!("f32_init_he.spv");
const INIT_UNIFORM_SHADER_F32: &[u8] = include_shader!("f32_init_uniform.spv");
const INIT_CONSTANT_SHADER: &[u8] = include_shader!("init_constant.spv");

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GPUMemoryOperation {
    Addition_F32,
    Subtract_F32,
    Multiply_F32,
    Divide_F32,
    Maximum_F32,
    Minimum_F32,
    ReLU_F32,
    Sigmoid_F32,
    Softmax_F32,
    Conv1D_F32,
    Conv2D_F32,
    Conv3D_F32,
    MatMul_F32,
    MatMul1D2D_F32,
    MatMul2D1D_F32,
    MatMul2D2D_F32,
    MatMul2D3D_F32,
    MatMul3D2D_F32,
    MatMul3D3D_F32,
    MatMul3D1D_F32,
    MatMul1D3D_F32,
    InitXavier_F32,
    InitHe_F32,
    InitUniform_F32,
    InitConstant,
}

impl GPUMemoryOperation {
    pub fn get_shader_bytes(&self) -> &[u8] {
        match self {
            GPUMemoryOperation::Addition_F32 => ADD_SHADER_F32,
            GPUMemoryOperation::Subtract_F32 => SUB_SHADER_F32,
            GPUMemoryOperation::Multiply_F32 => MUL_SHADER_F32,
            GPUMemoryOperation::Divide_F32 => DIV_SHADER_F32,
            GPUMemoryOperation::Maximum_F32 => MAX_SHADER_F32,
            GPUMemoryOperation::Minimum_F32 => MIN_SHADER_F32,
            GPUMemoryOperation::ReLU_F32 => RELU_SHADER_F32,
            GPUMemoryOperation::Sigmoid_F32 => SIGMOID_SHADER_F32,
            GPUMemoryOperation::Softmax_F32 => SOFTMAX_SHADER_F32,
            GPUMemoryOperation::Conv1D_F32 => CONV1D_SHADER_F32,
            GPUMemoryOperation::Conv2D_F32 => CONV2D_SHADER_F32,
            GPUMemoryOperation::Conv3D_F32 => CONV3D_SHADER_F32,
            GPUMemoryOperation::MatMul_F32 => MATMUL_SHADER_F32,
            GPUMemoryOperation::MatMul1D2D_F32 => MATMUL_1D_2D_SHADER_F32,
            GPUMemoryOperation::MatMul2D1D_F32 => MATMUL_2D_1D_SHADER_F32,
            GPUMemoryOperation::MatMul2D2D_F32 => MATMUL_2D_2D_SHADER_F32,
            GPUMemoryOperation::MatMul2D3D_F32 => MATMUL_2D_3D_SHADER_F32,
            GPUMemoryOperation::MatMul3D2D_F32 => MATMUL_3D_2D_SHADER_F32,
            GPUMemoryOperation::MatMul3D3D_F32 => MATMUL_3D_3D_SHADER_F32,
            GPUMemoryOperation::MatMul3D1D_F32 => MATMUL_3D_1D_SHADER_F32,
            GPUMemoryOperation::MatMul1D3D_F32 => MATMUL_1D_3D_SHADER_F32,
            GPUMemoryOperation::InitXavier_F32 => INIT_XAVIER_SHADER_F32,
            GPUMemoryOperation::InitHe_F32 => INIT_HE_SHADER_F32,
            GPUMemoryOperation::InitUniform_F32 => INIT_UNIFORM_SHADER_F32,
            GPUMemoryOperation::InitConstant => INIT_CONSTANT_SHADER,
        }
    }

    pub fn is_init(&self) -> bool {
        matches!(
            self,
            GPUMemoryOperation::InitXavier_F32
                | GPUMemoryOperation::InitHe_F32
                | GPUMemoryOperation::InitUniform_F32
                | GPUMemoryOperation::InitConstant
        )
    }

    pub fn expect_init(&self) {
        if !self.is_init() {
            panic!("Expected init GPUMemoryOperation, found {:?}", self);
        }
    }
}
