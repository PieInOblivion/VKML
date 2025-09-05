use crate::{
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::{fmt::{Debug, Formatter, Result as FmtResult}, sync::Arc};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

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
        let src_mem = tensor_graph.get_gpu_memory_or_panic(self.src);
        let weights_mem = tensor_graph.get_gpu_memory_or_panic(self.weights);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(self.dst);

        let src_tensor = tensor_graph.tensors.get(self.src).unwrap();
        let weights_tensor = tensor_graph.tensors.get(self.weights).unwrap();
        let dst_tensor = tensor_graph.tensors.get(self.dst).unwrap();

        let bias_mem = self
            .bias
            .as_ref()
            .map(|bias_id| tensor_graph.get_gpu_memory_or_panic(*bias_id));

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
                (in_height + 2 * self.padding.0 - filter_height) / self.stride.0 + 1;
            let expected_out_width =
                (in_width + 2 * self.padding.1 - filter_width) / self.stride.1 + 1;

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

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::Conv2D)
                .ok_or("Conv2D pipeline not found")?;

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
                gpu.get_compute_pipelines().get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const Conv2DPushConstants as *const u8,
                    std::mem::size_of::<Conv2DPushConstants>(),
                ),
            );

            // Calculate dispatch size based on output dimensions
            // Each thread computes one output element
            let total_output_elements = batch_size * out_channels * out_height * out_width;
            let workgroup_size = 256; // Match local_size_x from shader
            let num_workgroups: usize =
                (total_output_elements + workgroup_size - 1) / workgroup_size;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    fn execute_cpu(&self, tensor_graph: Arc<TensorGraph>) {
        let src_data = tensor_graph.tensors[self.src].data.read_data();
        let weights_data = tensor_graph.tensors[self.weights].data.read_data();
        let mut dst_data = tensor_graph.tensors[self.dst].data.write_data();

        // Check optional bias tensor
        let bias_data = if let Some(bias_id) = self.bias {
            Some(tensor_graph.tensors[bias_id].data.read_data())
        } else {
            None
        };

        let src_tensor = &tensor_graph.tensors[self.src];
        let weights_tensor = &tensor_graph.tensors[self.weights];
        let dst_tensor = &tensor_graph.tensors[self.dst];

        let src_dims = src_tensor.desc.to_dims();
        let weight_dims = weights_tensor.desc.to_dims();
        let dst_dims = dst_tensor.desc.to_dims();

        assert_eq!(src_dims.len(), 4, "Conv2D requires 4D input tensor");
        assert_eq!(weight_dims.len(), 4, "Conv2D requires 4D weight tensor");
        assert_eq!(dst_dims.len(), 4, "Conv2D requires 4D output tensor");

        let batch_size = src_dims[0];
        let in_channels = src_dims[1];
        let in_height = src_dims[2];
        let in_width = src_dims[3];

        let out_channels = weight_dims[0];
        let filter_in_channels = weight_dims[1];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];

        let out_height = dst_dims[2];
        let out_width = dst_dims[3];

        // Verify output dimensions
        assert_eq!(
            batch_size, dst_dims[0],
            "Batch size mismatch: input={}, output={}",
            batch_size, dst_dims[0]
        );
        assert_eq!(
            out_channels, dst_dims[1],
            "Output channel mismatch: filter={}, output={}",
            out_channels, dst_dims[1]
        );
        assert_eq!(
            in_channels, filter_in_channels,
            "Input channel mismatch: input={}, filter={}",
            in_channels, filter_in_channels
        );

        // Zero initialize result
        for val in dst_data.iter_mut() {
            *val = 0.0;
        }

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;

                        // Apply kernel
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    // Handle padding correctly
                                    let ih = (oh * self.stride.0 + kh).checked_sub(self.padding.0);
                                    let iw = (ow * self.stride.1 + kw).checked_sub(self.padding.1);

                                    // Only process valid input positions
                                    if let (Some(ih), Some(iw)) = (ih, iw) {
                                        if ih < in_height && iw < in_width {
                                            let in_idx = ((b * in_channels + ic) * in_height + ih)
                                                * in_width
                                                + iw;
                                            let w_idx = ((oc * in_channels + ic) * kernel_h + kh)
                                                * kernel_w
                                                + kw;

                                            sum += src_data[in_idx] * weights_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias) = &bias_data {
                            sum += bias[oc];
                        }

                        let out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        dst_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }
}
