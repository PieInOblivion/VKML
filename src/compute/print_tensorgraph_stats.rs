use crate::{
    compute::compute_manager::ComputeManager, scheduler::create_execution_plan, tensor::DeviceId,
    tensor_graph::TensorId,
};
use std::collections::HashSet;

pub fn print_tensor_flow(cm: &ComputeManager) {
    println!("\n=== TENSOR GRAPH VISUALISATION ===\n");

    let plan = match create_execution_plan(cm) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create execution plan: {:?}", e);
            return;
        }
    };

    println!("Execution Plan: {} chunks", plan.chunks.len());
    println!("{:-<100}", "");

    let mut produced_tensors = HashSet::new();
    produced_tensors.extend(cm.tensor_graph.input_tensor_ids.iter().cloned());

    // Add parameter tensors (tensors with no producers)
    for tensor_id in 0..cm.tensors.len() {
        if !cm.tensor_graph.input_tensor_ids.contains(&tensor_id) {
            let producers = cm.tensor_graph.get_tensor_producers(tensor_id);
            if producers.is_empty() {
                produced_tensors.insert(tensor_id);
            }
        }
    }

    for (chunk_idx, chunk) in plan.chunks.iter().enumerate() {
        let device_str = match &chunk.device {
            DeviceId::Cpu => "CPU".to_string(),
            DeviceId::Gpu(g) => format!("GPU {}", g),
        };

        let total_ops: usize = chunk.operation_layers.iter().map(|layer| layer.len()).sum();
        let layer_count = chunk.operation_layers.len();

        println!(
            "\nChunk {}: device={} ops={} layers={} preds={:?} deps={:?}",
            chunk_idx, device_str, total_ops, layer_count, chunk.predecessors, chunk.dependents
        );
        println!(
            "  initial_dep_count={} is_output={} needs_host_wait_fence={}",
            chunk.initial_dep_count,
            chunk.is_output,
            chunk.needs_host_wait_fence.is_some()
        );
        println!("{:-<100}", "");

        for (layer_idx, layer) in chunk.operation_layers.iter().enumerate() {
            if layer_count > 1 {
                println!("  === Layer {} ({} ops) ===", layer_idx, layer.len());
            }
            for &op_id in layer {
                let layer_id = cm.tensor_graph.operation_to_layer[op_id];

                let layer_name = cm
                    .model
                    .layers
                    .get(&layer_id)
                    .map(|layer| layer.layer.name())
                    .unwrap_or_else(|| "Unknown".to_string());

                let instruction = format!("{:?}", cm.tensor_graph.operations[op_id]);

                println!(
                    "  Operation {} (Layer {:?} - {})",
                    op_id, layer_id, layer_name
                );
                println!("  Instruction: {}", instruction);

                let inputs = cm.tensor_graph.get_operation_inputs(op_id);
                println!("  Inputs:");
                for input in inputs {
                    let tensor = cm.tensor_read(input);
                    let dtype = format!("{:?}", tensor.desc().data_type());
                    let shape = format!("{:?}", tensor.desc().dims());

                    let location = match tensor.device() {
                        DeviceId::Cpu => "CPU".to_string(),
                        DeviceId::Gpu(gpu_idx) => format!("GPU {}", gpu_idx),
                    };

                    let producers: String = cm
                        .tensor_graph
                        .get_tensor_producers(input)
                        .iter()
                        .map(|&op| format!("{}", op))
                        .collect::<Vec<_>>()
                        .join(", ");

                    println!(
                        "    Tensor {} - DType: {} - Shape: {} - Location: {} - Producers: {}",
                        input,
                        dtype,
                        shape,
                        location,
                        if producers.is_empty() {
                            "None".to_string()
                        } else {
                            producers
                        }
                    );
                }

                let outputs = cm.tensor_graph.get_operation_outputs(op_id);
                println!("  Outputs:");
                for output in outputs {
                    let tensor = cm.tensor_read(output);
                    let dtype = format!("{:?}", tensor.desc().data_type());
                    let shape = format!("{:?}", tensor.desc().dims());

                    let location = match tensor.device() {
                        DeviceId::Cpu => "CPU".to_string(),
                        DeviceId::Gpu(gpu_idx) => format!("GPU {}", gpu_idx),
                    };

                    let consumers: Vec<String> = cm
                        .tensor_graph
                        .get_tensor_consumers(output)
                        .iter()
                        .map(|&op| format!("{}", op))
                        .collect();

                    println!(
                        "    Tensor {} - DType: {} - Shape: {} - Location: {} - Consumers: {}",
                        output,
                        dtype,
                        shape,
                        location,
                        if consumers.is_empty() {
                            "None".to_string()
                        } else {
                            consumers.join(", ")
                        }
                    );

                    produced_tensors.insert(output);
                }

                println!();
            }
        }
    }

    // Sort layer IDs for consistent output
    let mut layer_ids: Vec<_> = cm.model.layers.keys().cloned().collect();
    layer_ids.sort();

    for layer_id in &layer_ids {
        if let Some(layer) = cm.model.layers.get(layer_id) {
            let layer_name = layer.layer.name();
            let layer_config = layer.layer.config_string().unwrap_or_default();

            println!(
                "Layer {}: {} {}",
                layer_id,
                layer_name,
                if !layer_config.is_empty() {
                    format!("({})", layer_config)
                } else {
                    String::new()
                }
            );

            println!("  Input Connections:");
            if layer.input_connections.is_empty() {
                println!("    None (Input Layer)");
            } else {
                for (idx, conn) in layer.input_connections.iter().enumerate() {
                    let source_id = conn.get_layerid();
                    let output_idx = conn.get_outputidx();

                    let source_name = cm
                        .model
                        .layers
                        .get(&source_id)
                        .map(|l| l.layer.name())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!(
                        "    Connection {}: From Layer {} ({}) Output {}",
                        idx, source_id, source_name, output_idx
                    );
                }
            }

            println!("  Output Connections:");
            if layer.output_connections.is_empty() {
                println!("    None (Output Layer)");
            } else {
                for (idx, conn) in layer.output_connections.iter().enumerate() {
                    let target_id = conn.get_layerid();
                    let input_idx = conn.get_outputidx();

                    let target_name = cm
                        .model
                        .layers
                        .get(&target_id)
                        .map(|l| l.layer.name())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!(
                        "    Connection {}: To Layer {} ({}) Input {}",
                        idx, target_id, target_name, input_idx
                    );
                }
            }

            println!("  Tensors:");
            let layer_tensors: Vec<TensorId> = (0..cm.tensors.len())
                .filter(|&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(*layer_id)))
                .collect();

            for tensor_id in layer_tensors {
                let tensor = cm.tensor_read(tensor_id);

                println!(
                    "    Tensor {}: Shape {:?}, Size: {}",
                    tensor_id,
                    tensor.desc().dims(),
                    cm.format_memory_mb(tensor.desc().size_in_bytes() as u64)
                );
            }

            println!();
        }
    }

    println!("\n=== TENSOR GRAPH SUMMARY ===\n");
    println!("Total Tensors: {}", cm.tensors.len());
    println!("Total Operations: {}", cm.tensor_graph.operations.len());
    println!("Input Tensors: {}", cm.tensor_graph.input_tensor_ids.len());
    println!(
        "Output Tensors: {}",
        cm.tensor_graph.output_tensor_ids.len()
    );
    println!("Execution Stages: {}", plan.chunks.len());

    let total_memory = cm.tensor_graph.memory_requirements;
    println!(
        "\nTotal Model Memory Requirements: {}",
        cm.format_memory_mb(total_memory as u64)
    );

    println!("\nMemory usage by device:");
    for (device, used, avail) in cm.get_memory_usage_summary() {
        println!("  {} - In use: {} - Available: {}", device, used, avail);
    }
}
