use std::sync::Arc;

use crate::{
    dataloader::error::VKMLEngineError, gpu::vk_gpu::GPU, layer::execution::LayerExecution, model::{graph_model::GraphModel, instruction::Instruction, layer_connection::LayerId, tensor_desc::TensorDesc, weight_init::WeightInit}, thread_pool::thread_pool::ThreadPool};

use super::{cpu_compute::CPUCompute, location::ComputeLocation, memory_tracker::MemoryTracker, print_model_stats};

pub enum DeviceLocation {
    CPU,
    GPU(usize)
}

pub struct ComputeTensor {
    pub desc: TensorDesc,
    pub location: ComputeLocation,
}

pub struct ExecutionStep {
    pub layer_id: LayerId,
    layer_exec: LayerExecution,
    input_tensors: Vec<TensorDesc>,
    pub output_tensors: Vec<TensorDesc>,
}

pub struct ComputeManager {
    gpus: Vec<GPU>,
    current_gpu_index: usize,
    cpu: CPUCompute,
    thread_pool: Arc<ThreadPool>,

    pub model: GraphModel,
    pub execution_pipeline: Vec<ExecutionStep>,
}

impl ComputeManager {
    pub fn new(model: GraphModel, thread_pool: Arc<ThreadPool>) -> Result<Self, VKMLEngineError> {
        let gpus = Self::available_gpus()?;
        Self::new_with(model, thread_pool, gpus, None)
    }

    pub fn new_with(mut model: GraphModel, thread_pool: Arc<ThreadPool>, gpus: Vec<GPU>, cpu_memory_limit_bytes: Option<u64>) -> Result<Self, VKMLEngineError> {
        // Verify the model if not already verified
        if model.verified.is_none() {
            model.verify()?;
        }

        let cpu = CPUCompute::new(cpu_memory_limit_bytes, thread_pool.clone());
        
        // Create initial compute manager without execution pipeline
        let mut manager = Self {
            gpus,
            current_gpu_index: 0,
            cpu,
            thread_pool,
            model,
            execution_pipeline: Vec::new(),
        };
        
        // Build the execution pipeline
        manager.build_execution_pipeline()?;
        
        // Calculate total memory requirements
        let total_memory = manager.calculate_total_memory_requirements();
        let total_available: u64 = manager.gpus.iter()
            .map(|gpu| gpu.available_memory())
            .sum::<u64>()
            + manager.cpu.memory_tracking.get_available();

        // Check if we have enough memory
        if total_memory > total_available {
            return Err(VKMLEngineError::OutOfMemory(
                format!("Model requires {} bytes but only {} available", 
                    total_memory, total_available)
            ));
        }
        
        // Allocate layers based on execution pipeline
        manager.allocate_execution_pipeline()?;
        
        Ok(manager)
    }

    fn build_execution_pipeline(&mut self) -> Result<(), VKMLEngineError> {     
        // Get the execution order from the verified model
        let execution_order = match &self.model.verified {
            Some(verified) => &verified.execution_order,
            None => return Err(VKMLEngineError::VulkanLoadError("Model not verified".into())),
        };
        
        self.execution_pipeline.clear();
        
        // Process each layer in execution order
        for &layer_id in execution_order {
            let layer = self.model.layers.get(&layer_id)
                .ok_or_else(|| VKMLEngineError::VulkanLoadError(
                    format!("Layer {} not found in model", layer_id)
                ))?;
            
            // Get input shapes from connected layers
            let input_shapes: Vec<TensorDesc> = layer.input_connections.iter()
                .map(|connection| {
                    let input_id = connection.get_layerid();
                    let output_idx = connection.get_outputidx();
                    
                    // Find the execution step that produced this input
                    self.execution_pipeline.iter()
                        .find(|step| step.layer_id == input_id)
                        .map(|step| {
                            // Access the specific output tensor by index
                            if output_idx < step.output_tensors.len() {
                                step.output_tensors[output_idx].clone()
                            } else {
                                // Fallback using first output (should never happen if verification passed)
                                step.output_tensors[0].clone()
                            }
                        })
                        .unwrap_or_else(|| {
                            panic!("Could not find execution step for layer {}", input_id);
                        })
                })
                .collect();
            
            let input_shape_refs: Vec<&TensorDesc> = input_shapes.iter().collect();
            
            // For input layers or layers with no inputs, create a default shape
            let input_shape = if input_shapes.is_empty() {
                match layer.layer.input_requirements().0 {
                    0 => {
                        // Input layer - create a default shape based on output features
                        TensorDesc::Matrix { 
                            rows: self.model.batch_size, 
                            cols: layer.layer.out_features() 
                        }
                    },
                    _ => {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Layer {} requires inputs but none were provided", layer_id)
                        ));
                    }
                }
            } else {
                // Use the first input shape
                input_shapes[0].clone()
            };
            
            // Build layer execution plan
            let mut layer_exec = layer.layer.build_layer_exec(self.model.batch_size, &input_shape)?;
            
            // Update the LoadInput instructions to include output_idx
            for instruction in &mut layer_exec.instructions {
                // First handle the case for LoadInput (which will be renamed to CopyInput)
                if let Instruction::CopyInput { layer_idx, dst, .. } = instruction {
                    // Get the connection for this input index
                    if *layer_idx < layer.input_connections.len() {
                        let layer_tensor_idx = layer.input_connections[*layer_idx].get_outputidx();
                        
                        // Replace with updated CopyInput instruction
                        *instruction = Instruction::CopyInput {
                            layer_idx: *layer_idx,
                            layer_tensor_idx,
                            dst: dst.clone(),
                        };
                    }
                }
                // Also handle any newly added ReadInput instructions
                else if let Instruction::ReadInput { layer_idx, dst, .. } = instruction {
                    // Get the connection for this input index
                    if *layer_idx < layer.input_connections.len() {
                        let layer_tensor_idx = layer.input_connections[*layer_idx].get_outputidx();
                        
                        // Replace with updated ReadInput instruction
                        *instruction = Instruction::ReadInput {
                            layer_idx: *layer_idx,
                            layer_tensor_idx,
                            dst: dst.clone(),
                        };
                    }
                }
            }
            
            // Calculate output shapes using the new method
            let output_shapes = layer.layer.output_shapes(self.model.batch_size, &input_shape_refs)?;
            
            if output_shapes.is_empty() {
                return Err(VKMLEngineError::VulkanLoadError(
                    format!("Layer {} returned no output shapes", layer_id)
                ));
            }
            
            // Use the first output shape as the primary output (backwards compatibility)
            let output_tensor = output_shapes[0].clone();
            
            // Add to execution pipeline with all output shapes
            self.execution_pipeline.push(ExecutionStep {
                layer_id,
                layer_exec,
                input_tensors: input_shapes,
                output_tensors: output_shapes,
            });
        }
        
        Ok(())
    }

    fn calculate_total_memory_requirements(&self) -> u64 {
        self.execution_pipeline.iter()
            .map(|step| {
                step.layer_exec.tensors.values()
                    .map(|tensor_info| tensor_info.desc.size_in_bytes() as u64)
                    .sum::<u64>()
            })
            .sum()
    }

    fn allocate_execution_pipeline(&mut self) -> Result<(), VKMLEngineError> {
        for step_index in 0..self.execution_pipeline.len() {
            // Get step information
            let step = &self.execution_pipeline[step_index];
            let layer_id = step.layer_id;
            
            // Find tensors that need allocation
            let tensors_to_allocate: Vec<(String, TensorDesc)> = step.layer_exec.tensors.iter()
                .filter(|(_, tensor)| matches!(tensor.location, ComputeLocation::Unallocated))
                .map(|(name, tensor)| (name.clone(), tensor.desc.clone()))
                .collect();
            
            // Skip if nothing to allocate
            if tensors_to_allocate.is_empty() {
                continue;
            }
            
            // Calculate total memory required
            let total_memory: u64 = tensors_to_allocate.iter()
                .map(|(_, desc)| desc.size_in_bytes() as u64)
                .sum();
            
            // Get weight initialization strategy
            let layer = self.model.layers.get_mut(&layer_id).unwrap();
            let weight_init = layer.weight_init.as_ref().unwrap_or(&self.model.weight_init).clone();
            
            // Find optimal device using the extracted method
            let target_device = self.find_optimal_device(total_memory)
                .ok_or_else(|| VKMLEngineError::OutOfMemory(
                    format!("No device has enough memory for layer {}: {} bytes", layer_id, total_memory)
                ))?;
            
            // Allocate each tensor using the extracted method
            for (name, desc) in tensors_to_allocate {
                let location = self.allocate_tensor(&desc, &target_device, &weight_init)?;
                
                // Update tensor location directly
                let tensor = &mut self.execution_pipeline[step_index].layer_exec.tensors.get_mut(&name).unwrap();
                tensor.location = location;
            }
        }
        
        Ok(())
    }

    pub fn available_gpus() -> Result<Vec<GPU>, VKMLEngineError> {
        let gpu_info = GPU::available_gpus()?;
        let mut gpus = Vec::with_capacity(gpu_info.len());
        
        for info in gpu_info {
            if let Ok(gpu) = GPU::new(info.device_index) {
                gpus.push(gpu);
            }
        }
        
        Ok(gpus)
    }

    fn find_optimal_device(&mut self, memory_required: u64) -> Option<DeviceLocation> {
        // Only check GPUs starting from our current one to maintain sequential blocks
        for idx in self.current_gpu_index..self.gpus.len() {
            if memory_required <= self.gpus[idx].available_memory() {
                self.current_gpu_index = idx;
                return Some(DeviceLocation::GPU(idx));
            }
        }

        // If no GPU has enough memory, try CPU
        if memory_required <= self.cpu.memory_tracking.get_available() {
            // Rust creates empy range when the start is equal or greater than end
            // Has one downside that if someone runs code with 18446744073709551615 GPUS on 64bit systems or 4294967295 on 32bit systems it will skip their last GPU :(
            self.current_gpu_index = usize::MAX;
            Some(DeviceLocation::CPU)
        } else {
            None
        }
    }

    fn allocate_tensor(&mut self, desc: &TensorDesc, target_device: &DeviceLocation, weight_init: &WeightInit) -> Result<ComputeLocation, VKMLEngineError> {
        // Allocate memory on target device
        let size_in_bytes = desc.size_in_bytes() as u64;

        // NOTE: There is probably a better place to put this
        // The most optimal value depends on each machine. This will serve as a general value for now
        // Constant type weight_init probably never needs to be multi-threaded?
        let parallel_threshold = 10000;

        let total_elements = desc.num_elements();
        
        let initial_data = {
            if total_elements < parallel_threshold {
                weight_init.init(desc, total_elements)
            } else {
                weight_init.par_init(desc, total_elements, parallel_threshold, self.thread_pool.clone())
            }
        };

        match *target_device {
            DeviceLocation::CPU => {
                // Allocate CPU memory
                self.cpu.memory_tracking.allocate(size_in_bytes)?;
                Ok(ComputeLocation::CPU(initial_data))
            },
            DeviceLocation::GPU(idx) => {
                // Allocate GPU memory
                let gpu = &mut self.gpus[idx];
                gpu.allocate_memory(size_in_bytes)?;
                
                let gpu_memory = gpu.move_to_gpu_as_f32(&initial_data)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                
                Ok(ComputeLocation::GPU {
                    gpu_idx: idx,
                    memory: gpu_memory,
                })
            }
        }
    }

    fn move_tensor_to_device(
        &self,
        tensor: &mut ComputeTensor,
        target_device: &DeviceLocation
    ) -> Result<(), VKMLEngineError> {
        // If tensor is Parameterless, no need to move anything
        if let ComputeLocation::Unallocated = tensor.location {
            return Ok(());
        }
    
        // If already on target device, nothing to do
        match (&tensor.location, target_device) {
            (ComputeLocation::CPU(_), DeviceLocation::CPU) => return Ok(()),
            (ComputeLocation::GPU { gpu_idx, .. }, DeviceLocation::GPU(target_idx))
                if gpu_idx == target_idx => return Ok(()),
            _ => {}
        }

        // Get size for memory tracking
        let size = tensor.desc.size_in_bytes() as u64;
    
        // Allocate on new device first
        match target_device {
            DeviceLocation::CPU => {
                self.cpu.memory_tracking.allocate(size)?;
            },
            DeviceLocation::GPU(idx) => {
                self.gpus[*idx].allocate_memory(size)?;
            }
        }
    
        // Perform the move
        let new_location = match &tensor.location {
            ComputeLocation::CPU(data) => {
                match target_device {
                    DeviceLocation::CPU => unreachable!(), // Handled above
                    DeviceLocation::GPU(idx) => {
                        // CPU to GPU
                        let gpu = &self.gpus[*idx];
                        let gpu_mem = gpu.move_to_gpu_as_f32(data)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
    
                        ComputeLocation::GPU {
                            gpu_idx: *idx,
                            memory: gpu_mem,
                        }
                    }
                }
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                match target_device {
                    DeviceLocation::CPU => {
                        // GPU to CPU
                        let gpu = &self.gpus[*gpu_idx];
                        let cpu_data = gpu.read_memory(memory)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        ComputeLocation::CPU(cpu_data)
                    },
                    DeviceLocation::GPU(target_idx) => {
                        // GPU to different GPU
                        let source_gpu = &self.gpus[*gpu_idx];
                        let cpu_data = source_gpu.read_memory(memory)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        let target_gpu = &self.gpus[*target_idx];
                        let new_gpu_mem = target_gpu.move_to_gpu_as_f32(&cpu_data)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        ComputeLocation::GPU {
                            gpu_idx: *target_idx,
                            memory: new_gpu_mem,
                        }
                    }
                }
            },
            ComputeLocation::Unallocated => unreachable!(),
        };
    
        // Deallocate from old device
        match &tensor.location {
            ComputeLocation::CPU(_) => {
                self.cpu.memory_tracking.deallocate(size);
            },
            ComputeLocation::GPU { gpu_idx, .. } => {
                self.gpus[*gpu_idx].deallocate_memory(size);
            },
            ComputeLocation::Unallocated => unreachable!(),
        }
    
        // Update tensor location
        tensor.location = new_location;
        Ok(())
    }

    pub fn get_tensor_data(&self, tensor: &ComputeTensor) -> Result<(Vec<f32>, Option<usize>), VKMLEngineError> {
        match &tensor.location {
            ComputeLocation::CPU(data) => {
                Ok((data.clone(), None))
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                self.gpus[*gpu_idx].read_memory(memory)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))
                    .map(|data| (data, Some(*gpu_idx)))
            },
            ComputeLocation::Unallocated => {
                Ok((vec![], None))
            },
        }
    }
    
    // Get device description for a tensor
    pub fn get_device_description(&self, tensor: &ComputeTensor) -> String {
        match &tensor.location {
            ComputeLocation::CPU(_) => "CPU".to_string(),
            ComputeLocation::GPU { gpu_idx, .. } => format!("GPU {}", gpu_idx),
            ComputeLocation::Unallocated => "Unallocated".to_string(),
        }
    }
    
    // Calculate parameters for a layer
    pub fn calculate_layer_parameters(&self, layer_id: LayerId) -> usize {
        if let Some(layer) = self.model.layers.get(&layer_id) {
            if let Some(step) = self.execution_pipeline.iter().find(|step| step.layer_id == layer_id) {
                if layer.layer.requires_parameters() {
                    let input_shapes: Vec<&TensorDesc> = step.input_tensors.iter().collect();
                    if let Some((weights, biases)) = layer.layer.parameter_shapes(&input_shapes) {
                        return weights.num_elements() + 
                               if layer.layer.to_string().contains("bias=false") { 0 } 
                               else { biases.num_elements() };
                    }
                }
            }
        }
        0
    }
    
    // Format memory size in MB
    pub fn format_memory_mb(&self, bytes: u64) -> String {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
    
    // Get total memory usage per device
    pub fn get_memory_usage_summary(&self) -> Vec<(String, String, String)> {
        let mut result = Vec::new();
        
        // CPU memory
        result.push((
            "CPU".to_string(),
            self.format_memory_mb(self.cpu.memory_tracking.get_current()),
            self.format_memory_mb(self.cpu.memory_tracking.get_available())
        ));
        
        // GPU memory
        for (i, gpu) in self.gpus.iter().enumerate() {
            result.push((
                format!("GPU {}", i),
                self.format_memory_mb(gpu.total_memory() - gpu.available_memory()),
                self.format_memory_mb(gpu.available_memory())
            ));
        }
        
        result
    }

    pub fn get_layer_output_shapes(&self, layer_id: LayerId) -> Option<Vec<&TensorDesc>> {
        self.execution_pipeline.iter()
            .find(|step| step.layer_id == layer_id)
            .map(|step| step.output_tensors.iter().collect())
    }
    
    // Get the memory usage for a layer
    pub fn get_layer_memory_usage(&self, layer_id: LayerId) -> u64 {
        self.execution_pipeline.iter()
            .find(|step| step.layer_id == layer_id)
            .map_or(0, |step| {
                step.layer_exec.tensors.values()
                    .map(|tensor| tensor.desc.size_in_bytes() as u64)
                    .sum()
            })
    }
    
    // Get the output tensor name for a layer
    pub fn get_layer_output_tensor_name(&self, layer_id: LayerId) -> Option<&str> {
        self.execution_pipeline.iter()
            .find(|step| step.layer_id == layer_id)
            .and_then(|step| step.layer_exec.outputs.first().map(|s| s.as_str()))
    }
    
    // Get the execution order
    pub fn get_execution_order_slice(&self) -> &[LayerId] {
        if let Some(verified) = &self.model.verified {
            &verified.execution_order
        } else {
            &[] // Empty slice literal
        }
    }
    // Access specific tensor for a layer (by name)
    pub fn get_layer_tensor(&self, layer_id: LayerId, tensor_name: &str) -> Option<&ComputeTensor> {
        self.execution_pipeline.iter()
            .find(|step| step.layer_id == layer_id)
            .and_then(|step| step.layer_exec.tensors.get(tensor_name))
    }
    
    // Get all tensor names for a layer
    pub fn get_layer_tensor_names(&self, layer_id: LayerId) -> Option<Vec<&str>> {
        self.execution_pipeline.iter()
            .find(|step| step.layer_id == layer_id)
            .map(|step| step.layer_exec.tensors.keys().map(|k| k.as_str()).collect())
    }

    pub fn print_model_stats(&self) {
        print_model_stats::print_model_stats(self);
    }

    pub fn print_layer_values(&self, layer_id: usize) -> Result<(), VKMLEngineError> {
        print_model_stats::print_layer_values(self, layer_id)
    }
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        // Clear layers first to ensure proper GPU memory cleanup
        self.execution_pipeline.clear();
    }
}
