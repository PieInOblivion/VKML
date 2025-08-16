use crate::{
    compute::compute_manager::DeviceLocation,
    instruction::{
        div_inplace::DivInplaceInstruction, max_inplace::MaxInplaceInstruction,
        min_inplace::MinInplaceInstruction, mul_inplace::MulInplaceInstruction,
        sub_inplace::SubInplaceInstruction,
    },
    tensor::tensor_desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};

use super::{
    add::AddInstruction, add_inplace::AddInplaceInstruction, concat::ConcatInstruction,
    conv2d::Conv2DInstruction, copy::CopyInstruction, div::DivInstruction, gelu::GELUInstruction,
    instruction::Instruction, leaky_relu::LeakyReLUInstruction, matmul::MatMulInstruction,
    max::MaxInstruction, min::MinInstruction, mul::MulInstruction, relu::ReLUInstruction,
    reshape::ReshapeInstruction, sigmoid::SigmoidInstruction, silu::SiLUInstruction,
    softmax::SoftmaxInstruction, sub::SubInstruction, tanh::TanhInstruction,
    transfer::TransferToDeviceInstruction,
};

/// Factory for creating instruction objects
pub struct Instructions;

impl Instructions {
    // Element-wise operations
    pub fn add(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(AddInstruction { src1, src2, dst })
    }

    pub fn sub(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(SubInstruction { src1, src2, dst })
    }

    pub fn mul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(MulInstruction { src1, src2, dst })
    }

    pub fn div(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(DivInstruction { src1, src2, dst })
    }

    pub fn max(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(MaxInstruction { src1, src2, dst })
    }

    pub fn min(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(MinInstruction { src1, src2, dst })
    }

    // Element-wise in place operations
    pub fn add_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(AddInplaceInstruction { dst, src1 })
    }

    pub fn sub_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(SubInplaceInstruction { dst, src1 })
    }

    pub fn mul_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(MulInplaceInstruction { dst, src1 })
    }

    pub fn div_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(DivInplaceInstruction { dst, src1 })
    }

    pub fn max_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(MaxInplaceInstruction { dst, src1 })
    }

    pub fn min_inplace(dst: TensorId, src1: TensorId) -> Box<dyn Instruction> {
        Box::new(MinInplaceInstruction { dst, src1 })
    }

    // Activation functions
    pub fn relu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(ReLUInstruction { src, dst })
    }

    pub fn leaky_relu(src: TensorId, dst: TensorId, alpha: f32) -> Box<dyn Instruction> {
        Box::new(LeakyReLUInstruction { src, dst, alpha })
    }

    pub fn sigmoid(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(SigmoidInstruction { src, dst })
    }

    pub fn softmax(src: TensorId, dst: TensorId, dim: usize) -> Box<dyn Instruction> {
        Box::new(SoftmaxInstruction { src, dst, dim })
    }

    pub fn tanh(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(TanhInstruction { src, dst })
    }

    pub fn gelu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(GELUInstruction { src, dst })
    }

    pub fn silu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(SiLUInstruction { src, dst })
    }

    // Matrix operations
    pub fn matmul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(MatMulInstruction { src1, src2, dst })
    }

    pub fn matmul_with(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(MatMulInstruction { src1, src2, dst })
    }

    // Convolution
    pub fn conv2d(
        src: TensorId,
        weights: TensorId,
        bias: Option<TensorId>,
        dst: TensorId,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Box<dyn Instruction> {
        Box::new(Conv2DInstruction {
            src,
            weights,
            bias,
            dst,
            stride,
            padding,
        })
    }

    // Data movement
    pub fn copy(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
        Box::new(CopyInstruction { src, dst })
    }

    pub fn transfer_to_device(
        src: TensorId,
        dst: TensorId,
        source_device: DeviceLocation,
        target_device: DeviceLocation,
    ) -> Box<dyn Instruction> {
        Box::new(TransferToDeviceInstruction {
            src,
            dst,
            source_device,
            target_device,
        })
    }

    // Data shaping
    pub fn reshape(src: TensorId, dst: TensorId, new_shape: TensorDesc) -> Box<dyn Instruction> {
        Box::new(ReshapeInstruction {
            src,
            dst,
            new_shape,
        })
    }

    pub fn concat(sources: Vec<TensorId>, dst: TensorId, dim: usize) -> Box<dyn Instruction> {
        Box::new(ConcatInstruction { sources, dst, dim })
    }
}
