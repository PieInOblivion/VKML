use crate::{
    instruction::{
        add::add::AddInstruction,
        concat::concat::ConcatInstruction,
        conv::conv::{AutoPad, ConvInstruction},
        div::div::DivInstruction,
        identity::identity::IdentityInstruction,
        init_constant::init_constant::InitConstantInstruction,
        init_he::init_he::InitHeInstruction,
        init_uniform::init_uniform::InitUniformInstruction,
        init_xavier::init_xavier::InitXavierInstruction,
        instruction::Instruction,
        matmul::matmul::MatMulInstruction,
        max::max::MaxInstruction,
        min::min::MinInstruction,
        mul::mul::MulInstruction,
        relu::relu::ReLUInstruction,
        reshape::reshape::ReshapeInstruction,
        sigmoid::sigmoid::SigmoidInstruction,
        softmax::softmax::SoftmaxInstruction,
        sub::sub::SubInstruction,
        transfer::transfer::TransferToDeviceInstruction,
    },
    tensor::tensor::DeviceId,
    tensor_graph::tensor_graph::TensorId,
};

pub mod add;
pub mod concat;
pub mod conv;
pub mod div;
pub mod gpu_operations;
pub mod identity;
pub mod init_constant;
pub mod init_he;
pub mod init_uniform;
pub mod init_xavier;
pub mod instruction;
pub mod matmul;
pub mod max;
pub mod maxpool;
pub mod min;
pub mod mul;
pub mod relu;
pub mod reshape;
pub mod sigmoid;
pub mod softmax;
pub mod sub;
pub mod transfer;

pub fn add(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(AddInstruction { src1, src2, dst })
}

pub fn concat(sources: Vec<TensorId>, dst: TensorId, dim: usize) -> Box<dyn Instruction> {
    Box::new(ConcatInstruction { sources, dst, dim })
}

pub fn conv(
    src: TensorId,
    weights: TensorId,
    bias: Option<TensorId>,
    dst: TensorId,
    auto_pad: AutoPad,
    dilations: Vec<usize>,
    group: i64,
    kernel_shape: Vec<usize>,
    pads: Vec<usize>,
    strides: Vec<usize>,
) -> Box<dyn Instruction> {
    Box::new(ConvInstruction {
        src,
        weights,
        bias,
        dst,
        auto_pad,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
    })
}

pub fn div(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(DivInstruction { src1, src2, dst })
}

pub fn identity(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(IdentityInstruction { src, dst })
}

pub fn init_constant(dst: TensorId, constant: Vec<u8>) -> Box<dyn Instruction> {
    Box::new(InitConstantInstruction { dst, constant })
}

pub fn init_he(dst: TensorId) -> Box<dyn Instruction> {
    Box::new(InitHeInstruction { dst })
}

pub fn init_uniform(dst: TensorId, min: f32, max: f32) -> Box<dyn Instruction> {
    Box::new(InitUniformInstruction { dst, min, max })
}

pub fn init_xavier(dst: TensorId) -> Box<dyn Instruction> {
    Box::new(InitXavierInstruction { dst })
}

pub fn matmul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MatMulInstruction { src1, src2, dst })
}

pub fn max(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MaxInstruction { src1, src2, dst })
}

pub fn maxpool(
    src: TensorId,
    dst: TensorId,
    auto_pad: AutoPad,
    dilations: Vec<usize>,
    kernel_shape: Vec<usize>,
    pads: Vec<usize>,
    strides: Vec<usize>,
    ceil_mode: bool,
) -> Box<dyn Instruction> {
    Box::new(crate::instruction::maxpool::maxpool::MaxPoolInstruction {
        src,
        dst,
        auto_pad,
        dilations,
        kernel_shape,
        pads,
        strides,
        ceil_mode,
    })
}

pub fn min(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MinInstruction { src1, src2, dst })
}

pub fn mul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MulInstruction { src1, src2, dst })
}

pub fn relu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(ReLUInstruction { src, dst })
}

pub fn reshape(
    src: TensorId,
    dst: TensorId,
    new_shape: Vec<i64>,
    allowzero: Option<i64>,
) -> Box<dyn Instruction> {
    Box::new(ReshapeInstruction {
        src,
        dst,
        shape_values: new_shape,
        allowzero,
    })
}

pub fn sigmoid(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SigmoidInstruction { src, dst })
}

pub fn softmax(src: TensorId, dst: TensorId, axis: Option<i64>) -> Box<dyn Instruction> {
    Box::new(SoftmaxInstruction { src, dst, axis })
}

pub fn sub(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SubInstruction { src1, src2, dst })
}

pub fn transfer(
    src: TensorId,
    dst: TensorId,
    source_device: DeviceId,
    target_device: DeviceId,
) -> Box<dyn Instruction> {
    Box::new(TransferToDeviceInstruction {
        src,
        dst,
        source_device,
        target_device,
    })
}
