use rand::distr::{Distribution, Uniform};
use std::f32::consts::PI;
use std::sync::Arc;

use std::ptr;
use vulkanalia::{vk, vk::DeviceV1_0};

use crate::dataloader::error::VKMLError;
use crate::gpu::compute_pipelines::GPUMemoryOperation;
use crate::gpu::gpu_memory::GPUMemory;
use crate::gpu::vk_gpu::GPU;
use crate::tensor::tensor_desc::TensorDesc;
use crate::thread_pool::thread_pool::ThreadPool;
use crate::thread_pool::worker::{DataPtrF32, WorkType};

#[derive(Clone)]
pub enum WeightInit {
    Xavier,
    He,
    LeCun,
    UniformRandom { min: f32, max: f32 },
    Constant(f32),
}

impl WeightInit {
    // Box-Muller transform to generate normal distribution
    pub fn normal_sample(mean: f32, std_dev: f32) -> f32 {
        let mut rng = rand::rng();
        let uniform = Uniform::new(0.0f32, 1.0);

        let u1 = uniform.unwrap().sample(&mut rng);
        let u2 = uniform.unwrap().sample(&mut rng);

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z
    }

    pub fn init(&self, shape: &TensorDesc, total_elements: usize) -> Vec<f32> {
        let (fan_in, fan_out) = shape.calculate_fan_in_out();

        match self {
            WeightInit::Xavier => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                let dist = Uniform::new(-limit, limit);
                let mut rng = rand::rng();
                (0..total_elements)
                    .map(|_| dist.unwrap().sample(&mut rng))
                    .collect()
            }

            WeightInit::He => {
                let std_dev = (2.0 / fan_in as f32).sqrt();
                (0..total_elements)
                    .map(|_| Self::normal_sample(0.0, std_dev))
                    .collect()
            }

            WeightInit::LeCun => {
                let std_dev = (1.0 / fan_in as f32).sqrt();
                (0..total_elements)
                    .map(|_| Self::normal_sample(0.0, std_dev))
                    .collect()
            }

            WeightInit::UniformRandom { min, max } => {
                let dist = Uniform::new(*min, *max);
                let mut rng = rand::rng();
                (0..total_elements)
                    .map(|_| dist.unwrap().sample(&mut rng))
                    .collect()
            }

            WeightInit::Constant(value) => {
                vec![*value; total_elements]
            }
        }
    }

    pub fn par_init(
        &self,
        shape: &TensorDesc,
        total_elements: usize,
        chunk_size: usize,
        thread_pool: Arc<ThreadPool>,
    ) -> Vec<f32> {
        let mut result = vec![0.0; total_elements];
        let (fan_in, fan_out) = shape.calculate_fan_in_out();
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        let data_ptr = DataPtrF32(result.as_mut_ptr());
        let mut work_items = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_elements);

            work_items.push(WorkType::WeightInitChunk {
                init_type: self.clone(),
                start_idx: start,
                end_idx: end,
                data_ptr,
                fan_in,
                fan_out,
            });
        }

        let work_batch = thread_pool.submit_batch(work_items);
        work_batch.wait();

        result
    }

    pub fn init_gpu(&self, shape: &TensorDesc, gpu: &GPU) -> Result<GPUMemory, VKMLError> {
        let total_elements = shape.num_elements();
        let (fan_in, fan_out) = shape.calculate_fan_in_out();

        // allocate uninitialised GPU memory
        let gpu_buffer = gpu
            .allocate_uninitialised_gpu_memory_f32(total_elements)
            .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

        // push constants for the compute shader
        let mut push_constants = [0.0f32; 32];
        push_constants[0] = total_elements as f32;
        push_constants[1] = fan_in as f32;
        push_constants[2] = fan_out as f32;
        push_constants[3] = rand::random::<u32>() as f32; // random seed

        // Set operation-specific parameters
        match self {
            WeightInit::Xavier => {
                push_constants[4] = 1.0; // default gain
            }
            WeightInit::He => {
                push_constants[4] = 2.0_f32.sqrt(); // default gain for ReLU
            }
            WeightInit::LeCun => {
                push_constants[4] = 1.0; // default gain
            }
            WeightInit::UniformRandom { min, max } => {
                push_constants[4] = *min;
                push_constants[5] = *max;
            }
            WeightInit::Constant(value) => {
                push_constants[4] = *value;
            }
        }

        let gpu_operation = match self {
            WeightInit::Xavier => GPUMemoryOperation::InitXavier,
            WeightInit::He => GPUMemoryOperation::InitHe,
            WeightInit::LeCun => GPUMemoryOperation::InitLeCun,
            WeightInit::UniformRandom { .. } => GPUMemoryOperation::InitUniform,
            WeightInit::Constant(_) => GPUMemoryOperation::InitConstant,
        };

        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                next: ptr::null(),
                command_pool: gpu.get_command_pool(),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
            };

            let command_buffer = gpu
                .get_device()
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?[0];

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: ptr::null(),
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let desc_alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: ptr::null(),
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = gpu
                .get_device()
                .allocate_descriptor_sets(&desc_alloc_info)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?[0];

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: gpu_buffer.buffer,
                offset: 0,
                range: gpu_buffer.size,
            };

            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                image_info: ptr::null(),
                buffer_info: &buffer_info,
                texel_buffer_view: ptr::null(),
            };

            gpu.get_device()
                .update_descriptor_sets(&[write_descriptor_set], &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(gpu_operation)
                .ok_or_else(|| {
                    VKMLError::VulkanLoadError(format!("{:?} pipeline not found", gpu_operation))
                })?;

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

            let push_constant_bytes: &[u8] = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 32]>(),
            );

            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_compute_pipelines().get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constant_bytes,
            );

            let workgroup_size = 256;
            let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device()
                .end_command_buffer(command_buffer)
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            gpu.submit_command_buffers_and_wait(&[command_buffer])
                .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

            gpu.get_device()
                .free_command_buffers(gpu.get_command_pool(), &[command_buffer]);
        }

        Ok(gpu_buffer)
    }
}
