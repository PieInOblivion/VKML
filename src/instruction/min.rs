use crate::{
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use onnx_extractor::DataType;
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct MinInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MinInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Min(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MinInstruction {
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

    fn create_command_buffer(
        &self,
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src1_read = tensor_graph.tensor_read(self.src1);
        let src1_mem = src1_read.get_gpu_memory_or_panic();
        let src2_read = tensor_graph.tensor_read(self.src2);
        let src2_mem = src2_read.get_gpu_memory_or_panic();
        let dst_read = tensor_graph.tensor_read(self.dst);
        let dst_mem = dst_read.get_gpu_memory_or_panic();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: std::ptr::null(),
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: std::ptr::null(),
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src1 buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src1_mem.buffer,
                    offset: 0,
                    range: src1_mem.size,
                },
                // src2 buffer (binding 1)
                vk::DescriptorBufferInfo {
                    buffer: src2_mem.buffer,
                    offset: 0,
                    range: src2_mem.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                // src1 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[0],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                // src2 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[2],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];

            gpu.get_device()
                .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::Minimum)
                .ok_or(format!(
                    "{:?} pipeline not found",
                    GPUMemoryOperation::Minimum
                ))?;

            gpu.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            gpu.get_device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_compute_pipelines().get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let workgroup_size = 256;
            let num_elements = dst_mem.size / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size as u64 - 1) / workgroup_size as u64;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &TensorGraph) {
        assert!(
            self.src1 != self.dst && self.src2 != self.dst,
            "Cannot use Min for in-place operation. Use MinInplace instead."
        );

        // Read shapes using short-lived read locks so we can later acquire a write lock on dst.
        let a = {
            let g = tensor_graph.tensor_read(self.src1);
            g.desc.to_dims()
        };
        let b = {
            let g = tensor_graph.tensor_read(self.src2);
            g.desc.to_dims()
        };
        let c = {
            let g = tensor_graph.tensor_read(self.dst);
            g.desc.to_dims()
        };

        let bc = TensorDesc::broadcast_shape(&a, &b)
            .expect(&format!("Can't broadcast {:?} vs {:?}", a, b));
        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(&a, &c);
        let sb = TensorDesc::broadcast_strides(&b, &c);

        // Acquire read locks for inputs and write lock for output
        let src1_guard = tensor_graph.tensor_read(self.src1);
        let src2_guard = tensor_graph.tensor_read(self.src2);
        let mut dst_guard = tensor_graph.tensor_write(self.dst);

        // Element count (number of elements, not bytes)
        let elem_count = dst_guard.desc.num_elements();

        let a_bytes = src1_guard.read();
        let b_bytes = src2_guard.read();

        // Helper to ensure buffer lengths
        let expect_len = |buf_len: usize, elem_size: usize| {
            let expected = elem_count * elem_size;
            assert_eq!(
                buf_len, expected,
                "Tensor byte length mismatch: {} != {}",
                buf_len, expected
            );
        };

        match dst_guard.desc.data_type() {
            DataType::Uint8 => {
                expect_len(a_bytes.len(), 1);
                expect_len(b_bytes.len(), 1);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa);
                    let off2 = TensorDesc::offset(&idxs, &sb);
                    out_buf[i] = a_bytes[off1].min(b_bytes[off2]);
                }
            }
            DataType::Int8 => {
                expect_len(a_bytes.len(), 1);
                expect_len(b_bytes.len(), 1);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa);
                    let off2 = TensorDesc::offset(&idxs, &sb);
                    let v1 = a_bytes[off1] as i8;
                    let v2 = b_bytes[off2] as i8;
                    out_buf[i] = v1.min(v2) as u8;
                }
            }
            DataType::Uint16 => {
                expect_len(a_bytes.len(), 2);
                expect_len(b_bytes.len(), 2);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 2;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 2;
                    let a_v = u16::from_le_bytes([a_bytes[off1], a_bytes[off1 + 1]]);
                    let b_v = u16::from_le_bytes([b_bytes[off2], b_bytes[off2 + 1]]);
                    let r = a_v.min(b_v).to_le_bytes();
                    out_buf[2 * i] = r[0];
                    out_buf[2 * i + 1] = r[1];
                }
            }
            DataType::Int16 => {
                expect_len(a_bytes.len(), 2);
                expect_len(b_bytes.len(), 2);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 2;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 2;
                    let a_v = i16::from_le_bytes([a_bytes[off1], a_bytes[off1 + 1]]);
                    let b_v = i16::from_le_bytes([b_bytes[off2], b_bytes[off2 + 1]]);
                    let r = a_v.min(b_v).to_le_bytes();
                    out_buf[2 * i] = r[0];
                    out_buf[2 * i + 1] = r[1];
                }
            }
            DataType::Uint32 => {
                expect_len(a_bytes.len(), 4);
                expect_len(b_bytes.len(), 4);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 4;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 4;
                    let a_v = u32::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                    ]);
                    let b_v = u32::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 4 * i;
                    out_buf[base] = r[0];
                    out_buf[base + 1] = r[1];
                    out_buf[base + 2] = r[2];
                    out_buf[base + 3] = r[3];
                }
            }
            DataType::Int32 => {
                expect_len(a_bytes.len(), 4);
                expect_len(b_bytes.len(), 4);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 4;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 4;
                    let a_v = i32::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                    ]);
                    let b_v = i32::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 4 * i;
                    out_buf[base] = r[0];
                    out_buf[base + 1] = r[1];
                    out_buf[base + 2] = r[2];
                    out_buf[base + 3] = r[3];
                }
            }
            DataType::Uint64 => {
                expect_len(a_bytes.len(), 8);
                expect_len(b_bytes.len(), 8);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 8;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 8;
                    let a_v = u64::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                        a_bytes[off1 + 4],
                        a_bytes[off1 + 5],
                        a_bytes[off1 + 6],
                        a_bytes[off1 + 7],
                    ]);
                    let b_v = u64::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                        b_bytes[off2 + 4],
                        b_bytes[off2 + 5],
                        b_bytes[off2 + 6],
                        b_bytes[off2 + 7],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 8 * i;
                    for j in 0..8 {
                        out_buf[base + j] = r[j];
                    }
                }
            }
            DataType::Int64 => {
                expect_len(a_bytes.len(), 8);
                expect_len(b_bytes.len(), 8);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 8;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 8;
                    let a_v = i64::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                        a_bytes[off1 + 4],
                        a_bytes[off1 + 5],
                        a_bytes[off1 + 6],
                        a_bytes[off1 + 7],
                    ]);
                    let b_v = i64::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                        b_bytes[off2 + 4],
                        b_bytes[off2 + 5],
                        b_bytes[off2 + 6],
                        b_bytes[off2 + 7],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 8 * i;
                    for j in 0..8 {
                        out_buf[base + j] = r[j];
                    }
                }
            }
            DataType::Float => {
                expect_len(a_bytes.len(), 4);
                expect_len(b_bytes.len(), 4);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 4;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 4;
                    let a_v = f32::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                    ]);
                    let b_v = f32::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 4 * i;
                    out_buf[base] = r[0];
                    out_buf[base + 1] = r[1];
                    out_buf[base + 2] = r[2];
                    out_buf[base + 3] = r[3];
                }
            }
            DataType::Double => {
                expect_len(a_bytes.len(), 8);
                expect_len(b_bytes.len(), 8);
                let out_buf = dst_guard.get_cpu_memory_mut_slice_or_panic();
                for i in 0..elem_count {
                    let idxs = TensorDesc::unravel(i, &c);
                    let off1 = TensorDesc::offset(&idxs, &sa) * 8;
                    let off2 = TensorDesc::offset(&idxs, &sb) * 8;
                    let a_v = f64::from_le_bytes([
                        a_bytes[off1],
                        a_bytes[off1 + 1],
                        a_bytes[off1 + 2],
                        a_bytes[off1 + 3],
                        a_bytes[off1 + 4],
                        a_bytes[off1 + 5],
                        a_bytes[off1 + 6],
                        a_bytes[off1 + 7],
                    ]);
                    let b_v = f64::from_le_bytes([
                        b_bytes[off2],
                        b_bytes[off2 + 1],
                        b_bytes[off2 + 2],
                        b_bytes[off2 + 3],
                        b_bytes[off2 + 4],
                        b_bytes[off2 + 5],
                        b_bytes[off2 + 6],
                        b_bytes[off2 + 7],
                    ]);
                    let r = a_v.min(b_v).to_le_bytes();
                    let base = 8 * i;
                    for j in 0..8 {
                        out_buf[base + j] = r[j];
                    }
                }
            }
            other => panic!("Min: unsupported DataType {:?}", other),
        }
    }
}
