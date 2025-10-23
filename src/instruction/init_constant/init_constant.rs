use crate::ComputeManager;
use crate::error::VKMLError;
use crate::instruction::init_constant::push_constants::InitConstantPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{gpu_operations::GPUOperation, instruction::Instruction},
    tensor_graph::TensorId,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct InitConstantInstruction {
    pub dst: TensorId,
    pub constant: Vec<u8>, // raw bytes, little endian
}

impl Debug for InitConstantInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "InitConstant(dst={}, constant_len={})",
            self.dst,
            self.constant.len()
        )
    }
}

impl Instruction for InitConstantInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, _new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Use generic constant-init compute shader that writes elements up to 8 bytes
        // Prepare CPU-side values (element size, total bytes/words, push constants)
        // Determine element size for this tensor and pass it to the GPU so the shader
        // can write using the correct stride/width. Panic if we don't know the size.
        let elem_size_usize = dst_tensor
            .desc
            .data_type()
            .size_in_bytes()
            .unwrap_or_else(|| {
                panic!(
                    "InitConstant record_into_command_buffer: unknown element size for DataType {:?}",
                    dst_tensor.desc.data_type()
                )
            });
        let elem_size = elem_size_usize as u32;

        // Validate element size (host-side defense). We support up to 8 bytes per element.
        if elem_size_usize == 0 || elem_size_usize > 8 {
            panic!(
                "InitConstant record_into_command_buffer: unsupported element size {} (must be 1..=8)",
                elem_size_usize
            );
        }

        // Compute total bytes from the tensor descriptor (preferred source of truth)
        let total_bytes_usize = dst_tensor.desc.size_in_bytes();
        if total_bytes_usize == 0 {
            // nothing to do
            return Ok(());
        }
        if total_bytes_usize > u32::MAX as usize {
            panic!(
                "InitConstant record_into_command_buffer: total bytes {} too large (must fit in u32)",
                total_bytes_usize
            );
        }

        let total_words = total_bytes_usize.div_ceil(4) as u32;

        // Build a little-endian u64 from the first up to 8 bytes of `self.constant`.
        // Use `.get()` with a default of 0 so GPU path is tolerant of shorter inputs.
        let mut value_u64: u64 = 0;
        for i in 0..8usize {
            let b = *self.constant.get(i).unwrap_or(&0) as u64;
            value_u64 |= b << (8 * i);
        }

        let push_constants = InitConstantPushConstants {
            elem_size,
            value_lo: (value_u64 & 0xFFFFFFFF) as u32,
            value_hi: (value_u64 >> 32) as u32,
        };

        let pc_bytes = as_bytes(&push_constants);

        // Choose operation and workgroup
        let gpu_op = GPUOperation::InitConstant;
        let local_size = gpu.optimal_workgroup_size_1d(total_words as u64);
        let binding_count = 1; // dst only

        // Bind pipeline first so push-descriptors are associated with the correct layout
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);

        // bind dst buffer at binding 0
        gpu.bind_storage_buffers(command_buffer, &[dst_mem]);

        gpu.bind_push_constants(command_buffer, binding_count, pc_bytes);

        gpu.dispatch(command_buffer, local_size, [total_words as u64, 1, 1]);

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let dst = cm.tensor_write(self.dst);
        let dtype = dst.desc.data_type();

        // Use DataType's helper to get element size in bytes; panic if unknown so we don't assume a size.
        let required_elem_bytes: usize = dtype.size_in_bytes().unwrap_or_else(|| {
            panic!(
                "InitConstant execute_cpu: unknown element size for DataType {:?}",
                dtype
            )
        });
        assert!(required_elem_bytes > 0);

        // Require that the provided constant bytes contain at least one element.
        if self.constant.len() < required_elem_bytes {
            panic!(
                "InitConstant execute_cpu: constant bytes length {} too small for DataType {:?} (need {} bytes)",
                self.constant.len(),
                dtype,
                required_elem_bytes
            );
        }

        // Use the first `required_elem_bytes` bytes as the element pattern (little-endian assumed).
        let pattern = &self.constant[0..required_elem_bytes];

        let out = dst.get_cpu_memory_mut_slice_or_panic();

        // Destination length must be a multiple of the element size.
        if !out.len().is_multiple_of(required_elem_bytes) {
            panic!(
                "InitConstant execute_cpu: destination length {} is not a multiple of element size {}",
                out.len(),
                required_elem_bytes
            );
        }

        for chunk in out.chunks_mut(required_elem_bytes) {
            chunk.copy_from_slice(pattern);
        }
    }
}
