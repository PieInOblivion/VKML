use crate::ComputeManager;
use crate::instruction::max::push_constants::MaxPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{gpu_operations::GPUOperation, instruction::Instruction, max::f32_cpu::f32_cpu},
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct MaxInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Max(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MaxInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src1, self.src2]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if new_inputs.len() >= 2 {
            self.src1 = new_inputs[0];
            self.src2 = new_inputs[1];
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src1_tensor = cm.tensor_read(self.src1);
        let src1_mem = src1_tensor.get_gpu_memory_or_panic();
        let src2_tensor = cm.tensor_read(self.src2);
        let src2_mem = src2_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        let src1_desc = &src1_tensor.desc;
        let src2_desc = &src2_tensor.desc;
        let dst_desc = &dst_tensor.desc;

        let src1_dims = src1_desc.dims();
        let src2_dims = src2_desc.dims();
        let dst_dims = dst_desc.dims();

        // Prepare push constant data using shared PushConstants

        let rank = dst_dims.len() as u32;
        assert!(
            rank <= 8,
            "Max: tensor rank {} exceeds maximum supported rank of 8",
            rank
        );

        let mut dims_arr = [0u32; 8];
        for (i, &d) in dst_dims.iter().enumerate().take(8) {
            dims_arr[i] = d as u32;
        }

        let broadcast_dims =
            TensorDesc::broadcast_shape(src1_dims, src2_dims).unwrap_or_else(|| {
                panic!(
                    "GPU Max: Can't broadcast {:?} vs {:?}",
                    src1_dims, src2_dims
                )
            });

        assert_eq!(
            broadcast_dims, dst_dims,
            "GPU Max: Broadcast shape {:?} != dst shape {:?}",
            broadcast_dims, dst_dims
        );

        let strides_a_usize = TensorDesc::broadcast_strides(src1_dims, dst_dims);
        let strides_b_usize = TensorDesc::broadcast_strides(src2_dims, dst_dims);

        let mut strides_a_arr = [0u32; 8];
        for (i, &s) in strides_a_usize.iter().enumerate().take(8) {
            strides_a_arr[i] = s as u32;
        }

        let mut strides_b_arr = [0u32; 8];
        for (i, &s) in strides_b_usize.iter().enumerate().take(8) {
            strides_b_arr[i] = s as u32;
        }

        let total_elements: u64 = dst_dims.iter().map(|d| *d as u64).product();

        let push_const_values = MaxPushConstants {
            rank,
            pad: 0,
            total: total_elements as u32,
            dims: dims_arr,
            strides_a: strides_a_arr,
            strides_b: strides_b_arr,
        };

        let push_constant_bytes = as_bytes(&push_const_values);
        // Choose operation and element size based on tensor DataType
        let op_datatype = dst_tensor.desc.data_type();
        let gpu_op = match op_datatype {
            DataType::Float => GPUOperation::Maximum_F32,
            _ => {
                return Err(format!("GPU Max unimplemented for DataType {:?}", op_datatype).into());
            }
        };

        // Choose 1D local workgroup size and bind pipeline/descriptors
        let local_size = gpu.optimal_workgroup_size_1d(total_elements);
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size);
        gpu.bind_storage_buffers(command_buffer, &[src1_mem, src2_mem, dst_mem]);

        // Push constants to the shader
        gpu.bind_push_constants(command_buffer, push_constant_bytes);

        let num_elements: u64 = dst_dims.iter().map(|d| *d as u64).product();

        gpu.dispatch(command_buffer, local_size, [num_elements, 1, 1]);

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src1 != self.dst && self.src2 != self.dst,
            "Cannot use Max for in-place operation. Use MaxInplace instead."
        );
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src1_tensor.desc.dims();
        let b = src2_tensor.desc.dims();
        let c = dst_tensor.desc.dims().to_vec();

        let bc = TensorDesc::broadcast_shape(a, b)
            .unwrap_or_else(|| panic!("Can't broadcast {:?} vs {:?}", a, b));
        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(a, &c);
        let sb = TensorDesc::broadcast_strides(b, &c);

        let op_datatype = dst_tensor.desc.data_type();

        let src1_bytes = src1_tensor.get_cpu_memory_slice_or_panic();
        let src2_bytes = src2_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match op_datatype {
            DataType::Float => f32_cpu(sa, sb, c, src1_bytes, src2_bytes, dst_ptr),
            _ => unimplemented!(
                "max.rs unimplemented cpu instruction for DataType {:?}",
                dst_tensor.desc.data_type()
            ),
        }
    }
}
