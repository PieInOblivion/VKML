use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::dataloader::data_batch::DataBatch;
use crate::gpu::gpu_memory::GPUMemory;
use crate::model::instruction::Instruction;
use crate::model::weight_init::WeightInit;
use crate::tensor::tensor_data::TensorData;
use crate::tensor_graph::shared_tensor_graph::{SharedGPU, SharedTensorGraph};
use crate::tensor_graph::tensor_graph::{OperationId, TensorId};
use ash::vk;
use image::ColorType;
use rand::distr::Uniform;
use rand::prelude::Distribution;

use super::thread_pool::ThreadPool;

#[derive(Copy, Clone)]
pub struct DataPtrU8(pub *mut u8);
unsafe impl Send for DataPtrU8 {}
unsafe impl Sync for DataPtrU8 {}

#[derive(Copy, Clone)]
pub struct DataPtrF32(pub *mut f32);
unsafe impl Send for DataPtrF32 {}
unsafe impl Sync for DataPtrF32 {}

pub enum WorkType {
    LoadImageBatch {
        batch_number: usize,
        paths: Vec<PathBuf>,
        image_total_bytes_per_batch: usize,
        image_bytes_per_image: usize,
        image_color_type: ColorType,
        batch_size: usize,
        thread_pool: Arc<ThreadPool>,
    },
    LoadSingleImage {
        path: PathBuf,
        start_idx: usize,
        end_idx: usize,
        data_ptr: DataPtrU8,
    },
    WeightInitChunk {
        init_type: WeightInit,
        start_idx: usize,
        end_idx: usize,
        data_ptr: DataPtrF32,
        fan_in: usize,
        fan_out: usize,
    },
    TensorOperations {
        operations: Vec<OperationId>,
        instructions: Vec<Instruction>,
        shared_gpu: Arc<SharedGPU>,
        shared_tensor_graph: Arc<SharedTensorGraph>,
    },
}

pub enum WorkResult {
    LoadImageBatch {
        batch_number: usize,
        batch: DataBatch,
    },
    LoadSingleImage,
    WeightInitChunk,
    TensorOperations,
}

#[derive(Clone)]
pub struct WorkFuture {
    pub state: Arc<(Mutex<Option<WorkResult>>, Condvar)>,
}

impl WorkFuture {
    pub fn new() -> Self {
        WorkFuture {
            state: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    pub fn wait(&self) {
        let (lock, cvar) = &*self.state;
        let mut result = lock.lock().unwrap();
        while result.is_none() {
            result = cvar.wait(result).unwrap();
        }
    }

    pub fn wait_and_take(self) -> WorkResult {
        let (lock, cvar) = &*self.state;
        let mut result = lock.lock().unwrap();
        while result.is_none() {
            result = cvar.wait(result).unwrap();
        }
        result.take().unwrap()
    }

    pub fn is_complete(&self) -> bool {
        self.state.0.lock().unwrap().is_some()
    }

    pub fn complete(&self, result: WorkResult) {
        let (lock, cvar) = &*self.state;
        *lock.lock().unwrap() = Some(result);
        cvar.notify_one();
    }
}

pub struct WorkQueue {
    pub queue: Mutex<VecDeque<WorkItem>>,
    pub items_count: AtomicUsize,
    pub condvar: Condvar,
}

impl WorkQueue {
    pub fn submit_work_item(&self, work_item: WorkItem) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(work_item);
        self.items_count.fetch_add(1, Ordering::SeqCst);
    }

    pub fn submit_work_batch(&self, work_items: Vec<WorkItem>) {
        let batch_size = work_items.len();
        {
            let mut queue = self.queue.lock().unwrap();
            queue.reserve(batch_size);
            for work_item in work_items {
                queue.push_back(work_item);
            }
            self.items_count.fetch_add(batch_size, Ordering::SeqCst);
        }
    }

    fn wait_and_get_next_work(&self) -> Option<WorkItem> {
        let mut queue = self.queue.lock().unwrap();
        while self.items_count.load(Ordering::SeqCst) == 0 {
            queue = self.condvar.wait(queue).unwrap();
        }

        self.try_pop_work_item(&mut queue)
    }

    fn try_get_work(&self) -> Option<WorkItem> {
        let mut queue = self.queue.lock().unwrap();
        self.try_pop_work_item(&mut queue)
    }

    fn try_pop_work_item(&self, queue: &mut VecDeque<WorkItem>) -> Option<WorkItem> {
        if self.items_count.load(Ordering::SeqCst) > 0 {
            let item = queue.pop_front();
            if item.is_some() {
                self.items_count.fetch_sub(1, Ordering::SeqCst);
            }
            item
        } else {
            None
        }
    }
}

pub struct WorkItem {
    pub work: WorkType,
    pub future: WorkFuture,
}

pub struct WorkFutureBatch {
    pub futures: Vec<WorkFuture>,
}

impl WorkFutureBatch {
    pub fn is_complete(&self) -> bool {
        self.futures.iter().all(|f| f.is_complete())
    }

    pub fn wait(self) -> Vec<WorkResult> {
        self.futures
            .into_iter()
            .map(|future| future.wait_and_take())
            .collect()
    }
}

pub struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    pub fn new(id: usize, work_queue: Arc<WorkQueue>) -> Worker {
        let thread = thread::spawn(move || {
            loop {
                if let Some(work_item) = work_queue.wait_and_get_next_work() {
                    Self::process_work(&work_queue, work_item);
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }

    fn process_work(work_queue: &Arc<WorkQueue>, work_item: WorkItem) {
        let result = match work_item.work {
            WorkType::LoadImageBatch {
                batch_number,
                paths,
                image_total_bytes_per_batch,
                image_bytes_per_image,
                image_color_type,
                batch_size,
                thread_pool,
            } => Self::load_image_batch(
                work_queue,
                batch_number,
                paths,
                image_total_bytes_per_batch,
                image_bytes_per_image,
                image_color_type,
                batch_size,
                thread_pool,
            ),
            WorkType::LoadSingleImage {
                path,
                start_idx,
                end_idx,
                data_ptr,
            } => Self::load_single_image(path, start_idx, end_idx, data_ptr),
            WorkType::WeightInitChunk {
                init_type,
                start_idx,
                end_idx,
                data_ptr,
                fan_in,
                fan_out,
            } => Self::generate_weight_init_chunk(
                init_type, start_idx, end_idx, data_ptr, fan_in, fan_out,
            ),
            WorkType::TensorOperations {
                operations,
                instructions,
                shared_gpu,
                shared_tensor_graph,
            } => {
                Self::execute_batch_work(operations, instructions, shared_gpu, shared_tensor_graph)
            }
        };

        work_item.future.complete(result);
    }

    fn load_image_batch(
        work_queue: &Arc<WorkQueue>,
        batch_number: usize,
        paths: Vec<PathBuf>,
        image_total_bytes_per_batch: usize,
        image_bytes_per_image: usize,
        image_color_type: ColorType,
        batch_size: usize,
        thread_pool: Arc<ThreadPool>,
    ) -> WorkResult {
        let mut batch = DataBatch {
            data: vec![0u8; image_total_bytes_per_batch].into_boxed_slice(),
            samples_in_batch: paths.len(),
            bytes_per_sample: image_bytes_per_image,
            format: image_color_type.into(),
            labels: None,
            batch_number,
        };

        let data_ptr = DataPtrU8(batch.data.as_mut_ptr());

        let work_items = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| {
                let start = idx * image_bytes_per_image;
                let end = start + image_bytes_per_image;

                WorkType::LoadSingleImage {
                    path: path.clone(),
                    start_idx: start,
                    end_idx: end,
                    data_ptr,
                }
            })
            .collect();

        let work_batch = thread_pool.submit_batch(work_items);

        // TODO: Try generalise the work while waiting pattern
        // Process other work while waiting for image to load
        while !work_batch.is_complete() {
            if let Some(other_work) = work_queue.try_get_work() {
                Self::process_work(work_queue, other_work);
            }
        }

        WorkResult::LoadImageBatch {
            batch_number,
            batch,
        }
    }

    fn load_single_image(
        path: PathBuf,
        start_idx: usize,
        end_idx: usize,
        data_ptr: DataPtrU8,
    ) -> WorkResult {
        let img = image::open(path).unwrap();
        let img_bytes = img.as_bytes();
        debug_assert_eq!(img_bytes.len(), end_idx - start_idx);

        // SAFETY: Each task has a unique slice range, so no overlapping writes
        // TODO: Benchmark how much faster this is instead of returning each image and having parent thread combine them
        unsafe {
            std::ptr::copy_nonoverlapping(
                img_bytes.as_ptr(),
                data_ptr.0.add(start_idx),
                end_idx - start_idx,
            );
        }

        WorkResult::LoadSingleImage
    }

    pub fn generate_weight_init_chunk(
        init_type: WeightInit,
        start_idx: usize,
        end_idx: usize,
        data_ptr: DataPtrF32,
        fan_in: usize,
        fan_out: usize,
    ) -> WorkResult {
        let mut rng = rand::rng();

        // SAFETY: Each task works on a unique slice range, so no overlapping writes
        unsafe {
            match init_type {
                WeightInit::Xavier => {
                    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                    let dist = Uniform::new(-limit, limit);
                    for i in start_idx..end_idx {
                        *data_ptr.0.add(i) = dist.unwrap().sample(&mut rng);
                    }
                }
                WeightInit::He => {
                    let std_dev = (2.0 / fan_in as f32).sqrt();
                    for i in start_idx..end_idx {
                        *data_ptr.0.add(i) = WeightInit::normal_sample(0.0, std_dev);
                    }
                }
                WeightInit::LeCun => {
                    let std_dev = (1.0 / fan_in as f32).sqrt();
                    for i in start_idx..end_idx {
                        *data_ptr.0.add(i) = WeightInit::normal_sample(0.0, std_dev);
                    }
                }
                WeightInit::UniformRandom { min, max } => {
                    let dist = Uniform::new(min, max);
                    for i in start_idx..end_idx {
                        *data_ptr.0.add(i) = dist.unwrap().sample(&mut rng);
                    }
                }
                WeightInit::Constant(value) => {
                    for i in start_idx..end_idx {
                        *data_ptr.0.add(i) = value;
                    }
                }
            }
        }
        WorkResult::WeightInitChunk
    }

    pub fn execute_batch_work(
        operations: Vec<OperationId>,
        instructions: Vec<Instruction>,
        shared_gpu: Arc<SharedGPU>,
        shared_tensor_graph: Arc<SharedTensorGraph>,
    ) -> WorkResult {
        // Safely dereference the raw pointers
        let tensor_graph = unsafe { &*shared_tensor_graph.tensor_graph };
        let gpu = shared_gpu.gpu.map(|gpu_ptr| unsafe { &*gpu_ptr });

        // Early return if no operations
        if operations.is_empty() {
            return WorkResult::TensorOperations;
        }

        // Check if we have a GPU to work with
        if let Some(gpu) = gpu {
            // GPU implementation

            unsafe {
                // Allocate command buffers for all operations
                let alloc_info = vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    command_pool: gpu.command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: operations.len() as u32,
                    _marker: std::marker::PhantomData,
                };

                let command_buffers = match gpu.device.allocate_command_buffers(&alloc_info) {
                    Ok(buffers) => buffers,
                    Err(e) => {
                        eprintln!("Failed to allocate command buffers: {}", e);
                        return WorkResult::TensorOperations;
                    }
                };

                // Verify that we received the exact number of command buffers we requested
                if command_buffers.len() != operations.len() {
                    eprintln!(
                        "Mismatch between allocated command buffers ({}) and operations ({})",
                        command_buffers.len(),
                        operations.len()
                    );
                    // Free the command buffers we did get
                    gpu.device
                        .free_command_buffers(gpu.command_pool, &command_buffers);
                    return WorkResult::TensorOperations;
                }

                // Record commands for each operation
                let mut valid_buffers = Vec::new();

                for i in 0..operations.len() {
                    let op_id = &operations[i];
                    let cmd_buffer = command_buffers[i];
                    let instruction = &instructions[i];
                    let layer_id = op_id.0;

                    // Record commands based on instruction type
                    let result = match instruction {
                        Instruction::Add { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_add_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::Sub { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_sub_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::Mul { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_mul_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::Div { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_div_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::Max { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_max_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::Min { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_mem = tensor_graph.get_gpu_memory_or_panic(&src1_id);
                            let src2_mem = tensor_graph.get_gpu_memory_or_panic(&src2_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_min_command_buffer(cmd_buffer, src1_mem, src2_mem, dst_mem)
                        }

                        Instruction::ReLU { src, dst } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_relu_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::LeakyReLU { src, dst, alpha } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_leaky_relu_command_buffer(
                                cmd_buffer, src_mem, dst_mem, *alpha,
                            )
                        }

                        Instruction::Sigmoid { src, dst } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_sigmoid_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::Softmax { src, dst, dim } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            // Get tensor shape for softmax
                            let tensor = tensor_graph.get_tensor_by_id_or_error(&src_id).unwrap();

                            gpu.create_softmax_command_buffer(
                                cmd_buffer,
                                src_mem,
                                dst_mem,
                                *dim,
                                &tensor.desc.to_dims(),
                            )
                        }

                        Instruction::Tanh { src, dst } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_tanh_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::GELU { src, dst } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_gelu_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::SiLU { src, dst } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_silu_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::Conv2D {
                            src,
                            weights,
                            bias,
                            dst,
                            stride: (stride_h, stride_w),
                            padding: (padding_h, padding_w),
                        } => {
                            let src_id = TensorId(layer_id, src.clone());
                            let weights_id = TensorId(layer_id, weights.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let weights_mem = tensor_graph.get_gpu_memory_or_panic(&weights_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            // Get tensor descriptors
                            let src_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&src_id).unwrap();

                            let weights_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&weights_id).unwrap();

                            let dst_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&dst_id).unwrap();

                            // Get optional bias memory
                            let bias_mem = bias.as_ref().and_then(|bias_name| {
                                let bias_id = TensorId(layer_id, bias_name.clone());
                                Some(tensor_graph.get_gpu_memory_or_panic(&bias_id))
                            });

                            // Call the improved Conv2D implementation
                            gpu.create_conv2d_command_buffer(
                                cmd_buffer,
                                src_mem,
                                weights_mem,
                                bias_mem,
                                dst_mem,
                                src_tensor,
                                weights_tensor,
                                dst_tensor,
                                *stride_h,
                                *stride_w,
                                *padding_h,
                                *padding_w,
                            )
                        }

                        Instruction::ReadInput { .. } => {
                            // ReadInput is just a logical reference - no actual computation needed
                            // The tensor connections are already established in the tensor graph
                            // This is essentially a no-op at execution time
                            continue;
                        }

                        Instruction::Copy { src, dst } => {
                            // Both tensors are in the same layer
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src_mem = tensor_graph.get_gpu_memory_or_panic(&src_id);
                            let dst_mem = tensor_graph.get_gpu_memory_or_panic(&dst_id);

                            gpu.create_copy_command_buffer(cmd_buffer, src_mem, dst_mem)
                        }

                        Instruction::TransferToDevice { src, dst, .. } => {
                            // Form the source and destination tensor IDs
                            let src_id = TensorId(layer_id, src.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            // Get the tensors from the tensor graph
                            let src_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&src_id).unwrap();
                            let dst_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&dst_id).unwrap();

                            // Get the data from the source tensor
                            let data = src_tensor.data.get_data().unwrap();

                            // Update the destination tensor with the data
                            dst_tensor.data.update_data(data).unwrap();

                            // No need to create a command buffer for CPU transfer
                            // We just skip the command buffer entirely and return success
                            Ok(())
                        }

                        Instruction::MatMul { src1, src2, dst } => {
                            let src1_id = TensorId(layer_id, src1.clone());
                            let src2_id = TensorId(layer_id, src2.clone());
                            let dst_id = TensorId(layer_id, dst.clone());

                            let src1_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&src1_id).unwrap();
                            let src2_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&src2_id).unwrap();
                            let dst_tensor =
                                tensor_graph.get_tensor_by_id_or_error(&dst_id).unwrap();

                            // Use the unified matmul implementation which will internally choose
                            // between specialised and generic implementations
                            gpu.create_matmul_command_buffer(
                                cmd_buffer,
                                src1_tensor,
                                src2_tensor,
                                dst_tensor,
                            )
                        }

                        // Other operations not implemented yet
                        _ => {
                            panic!(
                                "Instruction type {:?} not implemented for GPU execution",
                                instruction
                            );
                        }
                    };

                    // If command buffer was successfully recorded, add it to valid buffers
                    match result {
                        Ok(_) => {
                            valid_buffers.push(cmd_buffer);
                        }
                        Err(e) => eprintln!(
                            "Failed to record command buffer for operation {:?}: {}",
                            op_id, e
                        ),
                    }
                }

                // Submit all valid command buffers as a batch across all compute queues
                if !valid_buffers.is_empty() {
                    if let Err(e) = gpu.submit_command_buffers_and_wait(&valid_buffers) {
                        eprintln!("Failed to submit batch operations: {}", e);
                    }
                }

                gpu.device
                    .free_command_buffers(gpu.command_pool, &command_buffers);
            }
        } else {
            // CPU implementation (placeholder)
            eprintln!("CPU batch execution not implemented yet");
        }

        WorkResult::TensorOperations
    }
}
