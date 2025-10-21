use crate::gpu::vk_gpu::Gpu;
use crate::instruction::gpu_operations::GPUOperation;
use crate::instruction::shape::push_constants::ShapePushConstants;
use crate::tensor::TensorDesc;
use crate::utils::as_bytes;
use crate::{ComputeManager, instruction::Instruction, tensor_graph::TensorId};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct ShapeInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    // Optional slicing attributes
    pub start: Option<i64>,
    pub end: Option<i64>,
}

impl Debug for ShapeInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Shape(src={}, dst={}, start={:?}, end={:?})",
            self.src, self.dst, self.start, self.end
        )
    }
}

impl Instruction for ShapeInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    // We only implement CPU execution for Shape
    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Compute shape bytes on host, then upload to GPU via a host-visible staging buffer
        let src_desc = cm.tensor_read(self.src).desc.clone();
        let rank = src_desc.ndim() as i64;

        let mut start = self.start.unwrap_or(0);
        let mut end = self.end.unwrap_or(rank);

        if start < 0 {
            start += rank;
        }
        if end < 0 {
            end += rank;
        }

        if start < 0 {
            start = 0;
        }
        if start > rank {
            start = rank;
        }

        if end < 0 {
            end = 0;
        }
        if end > rank {
            end = rank;
        }

        let slice_len = if start >= end {
            0usize
        } else {
            (end - start) as usize
        };

        // Prepare push constants: split i64 dims into low/high 32-bit words
        let mut dims_lo = [0u32; 8];
        let mut dims_hi = [0u32; 8];
        for (i, &d) in src_desc.dims().iter().enumerate().take(8) {
            let bytes = d.to_le_bytes();
            dims_lo[i] = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            dims_hi[i] = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        }

        let pc = ShapePushConstants {
            slice_len: slice_len as u32,
            start: start as u32,
            pad: 0,
            dims_lo,
            dims_hi,
        };

        // For GPU path do not perform CPU writes; read the dst tensor and ensure it's GPU-backed
        let dst_t = cm.tensor_read(self.dst);
        let dst_mem = dst_t.get_gpu_memory_or_panic();

        // Respect DataType: Shape always writes int64 outputs on GPU
        let dst_dtype = dst_t.desc.data_type();
        let gpu_op = match dst_dtype {
            DataType::Int64 => GPUOperation::Shape_Write_I64,
            _ => {
                return Err(format!(
                    "GPU Shape unimplemented for dst DataType: {:?}, expected Int64",
                    dst_dtype
                )
                .into());
            }
        };
        let local_size = gpu.optimal_workgroup_size_1d(slice_len as u64);
        let binding_count = 1; // dst only

        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
        gpu.bind_storage_buffers(command_buffer, &[dst_mem]);

        let pc_bytes = as_bytes(&pc);
        gpu.bind_push_constants(command_buffer, binding_count, pc_bytes);

        // Dispatch with one work item per shape element
        gpu.dispatch(command_buffer, local_size, [slice_len as u64, 1, 1]);

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Read input tensor descriptor
        let src_desc = cm.tensor_read(self.src).desc.clone();
        let rank = src_desc.ndim() as i64;

        // Apply ONNX semantics for start/end
        let mut start = self.start.unwrap_or(0);
        let mut end = self.end.unwrap_or(rank);

        if start < 0 {
            start += rank;
        }
        if end < 0 {
            end += rank;
        }

        // Clamp
        if start < 0 {
            start = 0;
        }
        if start > rank {
            start = rank;
        }

        if end < 0 {
            end = 0;
        }
        if end > rank {
            end = rank;
        }

        // Determine slice
        let slice_len = if start >= end {
            0usize
        } else {
            (end - start) as usize
        };

        // Build output shape values
        let mut out_vals: Vec<i64> = Vec::with_capacity(slice_len);
        if slice_len > 0 {
            let dims = src_desc.dims();
            for i in start..end {
                let idx = i as usize;
                let v = *dims.get(idx).unwrap_or(&0);
                out_vals.push(v);
            }
        }

        // Update destination descriptor to 1D int64 tensor with length slice_len
        {
            let dst_t = cm.tensor_write(self.dst);
            dst_t.desc = TensorDesc::new(vec![slice_len as i64], DataType::Int64);

            // Write values into CPU buffer (little-endian)
            let dst_bytes = dst_t.get_cpu_memory_mut_slice_or_panic();
            // Ensure size matches expected
            let expected = slice_len.saturating_mul(8);
            if dst_bytes.len() != expected {
                panic!(
                    "Shape: destination buffer size {} does not match expected {}",
                    dst_bytes.len(),
                    expected
                );
            }

            let mut off = 0usize;
            for &val in &out_vals {
                let be = val.to_le_bytes();
                dst_bytes[off..off + 8].copy_from_slice(&be);
                off += 8;
            }
        }
    }
}
