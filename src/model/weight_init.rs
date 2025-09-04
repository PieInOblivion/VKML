use rand::distr::{Distribution, Uniform};
use std::f32::consts::PI;
use zero_pool::{ZeroPool, zp_define_task_fn};

use std::ptr;
use vulkanalia::{vk, vk::DeviceV1_0};

use crate::dataloader::error::VKMLError;
use crate::gpu::compute_pipelines::GPUMemoryOperation;
use crate::gpu::gpu_memory::GPUMemory;
use crate::gpu::vk_gpu::GPU;
use crate::tensor::desc::TensorDesc;
use onnx_extractor::DataType;

// TODO: Make WeightInit::Load happen before reaching WeightInit compute here
// Then make these branches !unreachable()

#[derive(Clone)]
pub enum WeightInit {
    Xavier,
    He,
    LeCun,
    UniformRandom { min: f32, max: f32 },
    Constant(f32),
    Load(Vec<u8>, DataType),
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

    pub fn init(&self, shape: &TensorDesc, total_elements: usize) -> (Vec<u8>, DataType) {
        let (fan_in, fan_out) = shape.calculate_fan_in_out();

        match self {
            WeightInit::Xavier => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                let dist = Uniform::new(-limit, limit);
                let mut rng = rand::rng();
                let mut out = Vec::with_capacity(total_elements * 4);
                for _ in 0..total_elements {
                    let v: f32 = dist.unwrap().sample(&mut rng);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (out, DataType::Float)
            }

            WeightInit::He => {
                let std_dev = (2.0 / fan_in as f32).sqrt();
                let mut out = Vec::with_capacity(total_elements * 4);
                for _ in 0..total_elements {
                    let v = Self::normal_sample(0.0, std_dev);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (out, DataType::Float)
            }

            WeightInit::LeCun => {
                let std_dev = (1.0 / fan_in as f32).sqrt();
                let mut out = Vec::with_capacity(total_elements * 4);
                for _ in 0..total_elements {
                    let v = Self::normal_sample(0.0, std_dev);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (out, DataType::Float)
            }

            WeightInit::UniformRandom { min, max } => {
                let dist = Uniform::new(*min, *max);
                let mut rng = rand::rng();
                let mut out = Vec::with_capacity(total_elements * 4);
                for _ in 0..total_elements {
                    let v: f32 = dist.unwrap().sample(&mut rng);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (out, DataType::Float)
            }

            WeightInit::Constant(value) => {
                let mut out = Vec::with_capacity(total_elements * 4);
                for _ in 0..total_elements {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                (out, DataType::Float)
            }
            // TODO: Have databatch not need cloning
            WeightInit::Load(data, datatype) => (data.to_vec(), *datatype),
        }
    }

    pub fn par_init(
        &self,
        shape: &TensorDesc,
        total_elements: usize,
        chunk_size: usize,
        thread_pool: &ZeroPool,
    ) -> (Vec<u8>, DataType) {
        // TODO: Have this not need cloning
        if let WeightInit::Load(data, datatype) = self {
            return (data.to_vec(), *datatype);
        }

        let (fan_in, fan_out) = shape.calculate_fan_in_out();
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;

        let mut batch = vec![0; total_elements * 4];

        let tasks: Vec<_> = (0..num_chunks)
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(total_elements);

                WeightInitChunkParams {
                    init_type: self.clone(),
                    start_idx: start,
                    end_idx: end,
                    data_ptr: batch.as_mut_ptr(),
                    fan_in,
                    fan_out,
                }
            })
            .collect();

        let future = thread_pool.submit_batch_uniform(weight_init_chunk_task, &tasks);
        future.wait();

        (batch, DataType::Float)
    }

    pub fn init_gpu(&self, shape: &TensorDesc, gpu: &GPU) -> Result<GPUMemory, VKMLError> {
        let total_elements = shape.num_elements();
        let (fan_in, fan_out) = shape.calculate_fan_in_out();

        // TODO: Should be fixed after the generalised gpu weight init pattern changes
        if let WeightInit::Load(_, _) = self {
            return Err(VKMLError::VulkanLoadError(
                "Load variant cannot be used with init_gpu".to_string(),
            ));
        }

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
            WeightInit::Load(_, _) => {
                // nothing to set for Load
            }
        }

        let gpu_operation = match self {
            WeightInit::Xavier => GPUMemoryOperation::InitXavier,
            WeightInit::He => GPUMemoryOperation::InitHe,
            WeightInit::LeCun => GPUMemoryOperation::InitLeCun,
            WeightInit::UniformRandom { .. } => GPUMemoryOperation::InitUniform,
            WeightInit::Constant(_) => GPUMemoryOperation::InitConstant,
            WeightInit::Load(_, _) => {
                return Err(VKMLError::VulkanLoadError(
                    "Load variant cannot be used with init_gpu".to_string(),
                ));
            }
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

struct WeightInitChunkParams {
    init_type: WeightInit,
    start_idx: usize,
    end_idx: usize,
    // pointer to the raw u8 buffer where f32 little-endian bytes will be written
    data_ptr: *mut u8,
    fan_in: usize,
    fan_out: usize,
}

zp_define_task_fn!(weight_init_chunk_task, WeightInitChunkParams, |params| {
    unsafe {
        match &params.init_type {
            WeightInit::Xavier => {
                let limit = (6.0 / (params.fan_in + params.fan_out) as f32).sqrt();
                let dist = Uniform::new(-limit, limit);
                let mut rng = rand::rng();
                for i in params.start_idx..params.end_idx {
                    let v: f32 = dist.unwrap().sample(&mut rng);
                    let bytes = v.to_le_bytes();
                    let base = i * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), params.data_ptr.add(base), 4);
                }
            }
            WeightInit::He => {
                let std_dev = (2.0 / params.fan_in as f32).sqrt();
                for i in params.start_idx..params.end_idx {
                    let v = WeightInit::normal_sample(0.0, std_dev);
                    let bytes = v.to_le_bytes();
                    let base = i * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), params.data_ptr.add(base), 4);
                }
            }
            WeightInit::LeCun => {
                let std_dev = (1.0 / params.fan_in as f32).sqrt();
                for i in params.start_idx..params.end_idx {
                    let v = WeightInit::normal_sample(0.0, std_dev);
                    let bytes = v.to_le_bytes();
                    let base = i * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), params.data_ptr.add(base), 4);
                }
            }
            WeightInit::UniformRandom { min, max } => {
                let dist = Uniform::new(*min, *max);
                let mut rng = rand::rng();
                for i in params.start_idx..params.end_idx {
                    let v: f32 = dist.unwrap().sample(&mut rng);
                    let bytes = v.to_le_bytes();
                    let base = i * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), params.data_ptr.add(base), 4);
                }
            }
            WeightInit::Constant(value) => {
                let bytes = value.to_le_bytes();
                for i in params.start_idx..params.end_idx {
                    let base = i * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), params.data_ptr.add(base), 4);
                }
            }
            WeightInit::Load(data, _) => {
                let start_byte = params.start_idx * 4;
                let end_byte = params.end_idx * 4;
                let src_len = data.len();
                let copy_end = end_byte.min(src_len);
                if start_byte < copy_end {
                    let src = data.as_ptr().add(start_byte);
                    let dst = params.data_ptr.add(start_byte);
                    std::ptr::copy_nonoverlapping(src, dst, copy_end - start_byte);
                }
            }
        }
    }
});
