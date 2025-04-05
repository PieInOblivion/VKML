use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::dataloader::data_batch::DataBatch;
use crate::instruction::instruction::Instruction;
use crate::model::weight_init::WeightInit;
use crate::tensor_graph::shared_tensor_graph::{SharedGPU, SharedTensorGraph};
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
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
    GpuBatchOperations {
        operations: Vec<OperationId>,
        instruction_indices: Vec<usize>,
        shared_gpu: Arc<SharedGPU>,
        shared_tensor_graph: Arc<SharedTensorGraph>,
    },
    SingleCpuOperation {
        operation_id: OperationId,
        instruction_idx: usize,
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
            WorkType::GpuBatchOperations {
                operations,
                instruction_indices,
                shared_gpu,
                shared_tensor_graph,
            } => Self::execute_gpu_batch_work(
                operations,
                instruction_indices,
                shared_gpu,
                shared_tensor_graph,
            ),

            WorkType::SingleCpuOperation {
                operation_id,
                instruction_idx,
                shared_tensor_graph,
            } => Self::execute_single_cpu_operation(
                operation_id,
                instruction_idx,
                shared_tensor_graph,
            ),
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

    pub fn execute_gpu_batch_work(
        operations: Vec<OperationId>,
        instruction_indices: Vec<usize>,
        shared_gpu: Arc<SharedGPU>,
        shared_tensor_graph: Arc<SharedTensorGraph>,
    ) -> WorkResult {
        // Skip if no GPU or no operations
        let Some(gpu) = shared_gpu.gpu.map(|gpu_ptr| unsafe { &*gpu_ptr }) else {
            println!("No GPU given to gpu function");
            return WorkResult::TensorOperations;
        };

        if operations.is_empty() {
            return WorkResult::TensorOperations;
        }

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

            // Verify that we received the correct number of command buffers
            if command_buffers.len() != operations.len() {
                eprintln!(
                    "Mismatch between allocated command buffers ({}) and operations ({})",
                    command_buffers.len(),
                    operations.len()
                );
                gpu.device
                    .free_command_buffers(gpu.command_pool, &command_buffers);
                return WorkResult::TensorOperations;
            }

            // Record commands for each operation
            let mut valid_buffers = Vec::new();
            let tensor_graph = &*shared_tensor_graph.tensor_graph;

            for i in 0..operations.len() {
                let instruction = tensor_graph.get_instruction_or_panic(instruction_indices[i]);

                // Use the instruction's command buffer creation method
                match instruction.create_command_buffer(gpu, command_buffers[i], tensor_graph) {
                    Ok(_) => {
                        valid_buffers.push(command_buffers[i]);
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to record command buffer for operation {:?}: {}",
                            &operations[i], e
                        );
                    }
                }
            }

            // Submit all valid command buffers
            if !valid_buffers.is_empty() {
                if let Err(e) = gpu.submit_command_buffers_and_wait(&valid_buffers) {
                    eprintln!("Failed to submit batch operations: {}", e);
                }
            }

            gpu.device
                .free_command_buffers(gpu.command_pool, &command_buffers);
        }

        WorkResult::TensorOperations
    }

    // Specialized CPU single operation function
    pub fn execute_single_cpu_operation(
        operation_id: OperationId,
        instruction_idx: usize,
        shared_tensor_graph: Arc<SharedTensorGraph>,
    ) -> WorkResult {
        unsafe {
            // Get the tensor graph pointer
            let tensor_graph_ptr = shared_tensor_graph.tensor_graph as *mut TensorGraph;

            // Check if the instruction exists
            if instruction_idx >= (*tensor_graph_ptr).operations.len() {
                eprintln!("Invalid instruction index: {}", instruction_idx);
                return WorkResult::TensorOperations;
            }

            // Get a pointer to the specific instruction
            let instruction_ptr =
                &(*tensor_graph_ptr).operations[instruction_idx] as *const Box<dyn Instruction>;

            // Execute with separate pointers
            if let Err(e) = (*instruction_ptr).execute_cpu(&mut *tensor_graph_ptr) {
                eprintln!("CPU execution failed for operation {}: {}", operation_id, e);
            }
        }

        WorkResult::TensorOperations
    }
}
