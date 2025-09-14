use crate::{
    gpu::vk_gpu::GPU,
    instruction::{
        conv2d::f32_cpu::f32_cpu, gpu_operations::GPUMemoryOperation, instruction::Instruction,
    },
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct Conv2DInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Debug for Conv2DInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Conv2D(src={}, weights={}, bias={:?}, dst={}, stride={:?}, padding={:?})",
            self.src, self.weights, self.bias, self.dst, self.stride, self.padding
        )
    }
}

impl Instruction for Conv2DInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        let mut inputs = vec![self.src, self.weights];
        if let Some(bias) = self.bias {
            inputs.push(bias);
        }
        inputs
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if new_inputs.len() > 1 {
            self.weights = new_inputs[1];
        }

        if new_inputs.len() > 2 && self.bias.is_some() {
            self.bias = Some(new_inputs[2]);
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
        // Acquire read guards for tensors so we can access descriptors and GPU memory
        let src_tensor = tensor_graph.tensor_read(self.src);
        let weights_tensor = tensor_graph.tensor_read(self.weights);
        let dst_tensor = tensor_graph.tensor_read(self.dst);

        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let weights_mem = weights_tensor.get_gpu_memory_or_panic();
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Optional bias read guard (kept in scope)
        let bias_tensor_opt = if let Some(bid) = self.bias {
            Some(tensor_graph.tensor_read(bid))
        } else {
            None
        };

        let bias_mem = bias_tensor_opt
            .as_ref()
            .map(|t| t.get_gpu_memory_or_panic());

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

            // Get dimensions from tensor descriptors
            let src_dims = src_tensor.desc.to_dims();
            let filter_dims = weights_tensor.desc.to_dims();
            let dst_dims = dst_tensor.desc.to_dims();

            // Get strides for input, filter, and output tensors
            let src_strides = src_tensor.desc.strides();
            let filter_strides = weights_tensor.desc.strides();
            let dst_strides = dst_tensor.desc.strides();

            // Validate tensor dimensions
            if src_dims.len() != 4 || filter_dims.len() != 4 || dst_dims.len() != 4 {
                return Err("Conv2D requires 4D tensors for input, filters, and output".into());
            }

            let batch_size = src_dims[0];
            let in_channels = src_dims[1];
            let in_height = src_dims[2];
            let in_width = src_dims[3];

            let out_channels = filter_dims[0];
            let filter_in_channels = filter_dims[1];
            let filter_height = filter_dims[2];
            let filter_width = filter_dims[3];

            let out_batch = dst_dims[0];
            let out_channels_check = dst_dims[1];
            let out_height = dst_dims[2];
            let out_width = dst_dims[3];

            // Validation checks
            if batch_size != out_batch {
                return Err(format!(
                    "Batch size mismatch: input={}, output={}",
                    batch_size, out_batch
                )
                .into());
            }

            if out_channels != out_channels_check {
                return Err(format!(
                    "Output channel mismatch: filter={}, output={}",
                    out_channels, out_channels_check
                )
                .into());
            }

            if in_channels != filter_in_channels {
                return Err(format!(
                    "Input channel mismatch: input={}, filter={}",
                    in_channels, filter_in_channels
                )
                .into());
            }

            // Verify output dimensions match expected conv2d output size
            let expected_out_height =
                (in_height + 2 * self.padding.0 as i64 - filter_height) / self.stride.0 as i64 + 1;
            let expected_out_width =
                (in_width + 2 * self.padding.1 as i64 - filter_width) / self.stride.1 as i64 + 1;

            if out_height != expected_out_height || out_width != expected_out_width {
                return Err(format!(
                    "Output dimensions mismatch. Expected: {}×{}, Got: {}×{}",
                    expected_out_height, expected_out_width, out_height, out_width
                )
                .into());
            }

            // Update descriptor set with input, filter, bias (optional), and output buffers
            let buffer_infos = vec![
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src_mem.buffer,
                    offset: 0,
                    range: src_mem.size,
                },
                // filter buffer (binding 1)
                vk::DescriptorBufferInfo {
                    buffer: weights_mem.buffer,
                    offset: 0,
                    range: weights_mem.size,
                },
                // bias buffer (binding 2, optional)
                vk::DescriptorBufferInfo {
                    buffer: bias_mem.map_or(weights_mem.buffer, |b| b.buffer), // Reuse filter buffer if no bias
                    offset: 0,
                    range: bias_mem.map_or(4, |b| b.size), // Min size if no bias
                },
                // dst buffer (binding 3)
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = vec![
                // Input buffer descriptor
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
                // Filter buffer descriptor
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
                // Bias buffer descriptor
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
                // Output buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[3],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];

            gpu.get_device()
                .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

            let op_datatype = dst_tensor.desc.data_type();
            let gpu_op = match op_datatype {
                DataType::Float => GPUMemoryOperation::Conv2D_F32,
                _ => {
                    return Err(
                        format!("GPU Conv2D unimplemented for DataType {:?}", op_datatype).into(),
                    );
                }
            };

            // Use GPU pipeline from GPU helper
            let pipeline = gpu.get_or_create_pipeline(gpu_op);

            gpu.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            gpu.get_device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            // Create push constants
            #[repr(C)]
            struct Conv2DPushConstants {
                // Dimensions
                batch_size: u32,
                in_channels: u32,
                in_height: u32,
                in_width: u32,

                filter_out_channels: u32,
                filter_height: u32,
                filter_width: u32,

                out_height: u32,
                out_width: u32,

                // Convolution parameters
                stride_h: u32,
                stride_w: u32,
                padding_h: u32,
                padding_w: u32,

                // Tensor strides (up to 8 values, 4 for each tensor)
                src_stride_0: u32,
                src_stride_1: u32,
                src_stride_2: u32,
                src_stride_3: u32,

                filter_stride_0: u32,
                filter_stride_1: u32,
                filter_stride_2: u32,
                filter_stride_3: u32,

                dst_stride_0: u32,
                dst_stride_1: u32,
                dst_stride_2: u32,
                dst_stride_3: u32,

                use_bias: u32,
            }

            let push_constants = Conv2DPushConstants {
                batch_size: batch_size as u32,
                in_channels: in_channels as u32,
                in_height: in_height as u32,
                in_width: in_width as u32,

                filter_out_channels: out_channels as u32,
                filter_height: filter_height as u32,
                filter_width: filter_width as u32,

                out_height: out_height as u32,
                out_width: out_width as u32,

                stride_h: self.stride.0 as u32,
                stride_w: self.stride.1 as u32,
                padding_h: self.padding.0 as u32,
                padding_w: self.padding.1 as u32,

                // Input tensor strides
                src_stride_0: src_strides[0] as u32,
                src_stride_1: src_strides[1] as u32,
                src_stride_2: src_strides[2] as u32,
                src_stride_3: src_strides[3] as u32,

                // Filter tensor strides
                filter_stride_0: filter_strides[0] as u32,
                filter_stride_1: filter_strides[1] as u32,
                filter_stride_2: filter_strides[2] as u32,
                filter_stride_3: filter_strides[3] as u32,

                // Output tensor strides
                dst_stride_0: dst_strides[0] as u32,
                dst_stride_1: dst_strides[1] as u32,
                dst_stride_2: dst_strides[2] as u32,
                dst_stride_3: dst_strides[3] as u32,

                use_bias: if bias_mem.is_some() { 1 } else { 0 },
            };

            // Push constants to the shader
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const Conv2DPushConstants as *const u8,
                    std::mem::size_of::<Conv2DPushConstants>(),
                ),
            );

            // Calculate dispatch size based on output dimensions
            // Each thread computes one output element
            let total_output_elements: usize = (batch_size as usize)
                * (out_channels as usize)
                * (out_height as usize)
                * (out_width as usize);
            let workgroup_size = 256usize; // Match local_size_x from shader
            let num_workgroups: usize =
                (total_output_elements + workgroup_size - 1) / workgroup_size;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    fn execute_cpu(&self, tensor_graph: &TensorGraph) {
        let src_guard = tensor_graph.tensor_read(self.src);
        let weights_guard = tensor_graph.tensor_read(self.weights);
        let src_bytes = src_guard.get_cpu_memory_slice_or_panic().to_vec();
        let weight_bytes = weights_guard.get_cpu_memory_slice_or_panic().to_vec();
        let src_dims_i64 = src_guard.desc.to_dims();
        let weight_dims_i64 = weights_guard.desc.to_dims();

        let bias_bytes_vec_opt: Option<Vec<u8>> = self.bias.map(|b| {
            tensor_graph
                .tensor_read(b)
                .get_cpu_memory_slice_or_panic()
                .to_vec()
        });

        drop(src_guard);
        drop(weights_guard);

        // Read dst descriptor info before taking mutable write guard to avoid borrow conflicts
        let dst_read = tensor_graph.tensor_read(self.dst);
        let dst_data_type = dst_read.desc.data_type();
        let dst_dims_i64 = dst_read.desc.to_dims();
        drop(dst_read);

        let mut dst_guard = tensor_graph.tensor_write(self.dst);
        let dst_bytes = dst_guard.get_cpu_memory_mut_slice_or_panic();

        let src_dims = src_dims_i64.iter().map(|d| *d as usize).collect();
        let weight_dims = weight_dims_i64.iter().map(|d| *d as usize).collect();
        let dst_dims = dst_dims_i64.iter().map(|d| *d as usize).collect();

        let bias_bytes_ref_opt = bias_bytes_vec_opt.as_ref().map(|v| v.as_slice());

        match dst_data_type {
            onnx_extractor::DataType::Float => {
                f32_cpu(
                    src_dims,
                    weight_dims,
                    dst_dims,
                    src_bytes.as_slice(),
                    weight_bytes.as_slice(),
                    bias_bytes_ref_opt,
                    dst_bytes,
                    self.stride,
                    self.padding,
                );
            }
            other => panic!(
                "conv2d.rs execute_cpu: unimplemented CPU Conv2D for DataType {:?}",
                other
            ),
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }
}
