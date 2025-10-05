use crate::{
    compute::compute_manager::ComputeManager, tensor::tensor::DeviceId,
    tensor_graph::tensor_graph::TensorId,
};
use std::collections::HashSet;

pub fn print_tensor_flow(cm: &ComputeManager) {
    println!("\n=== TENSOR GRAPH VISUALIZATION ===\n");

    let execution_plan = cm.tensor_graph.create_stage_plan();

    println!("Execution Plan: {} stages", execution_plan.len());
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

    for (stage_idx, operations) in execution_plan.iter().enumerate() {
        println!("\nStage {}: {} operations", stage_idx + 1, operations.len());
        println!("{:-<100}", "");

        for op_id in operations {
            let layer_id = cm.tensor_graph.operation_to_layer[*op_id];

            // Fetch layer name using the correct layer_id
            let layer_name = cm
                .model
                .layers
                .get(&layer_id)
                .map(|layer| layer.layer.name())
                .unwrap_or_else(|| "Unknown".to_string());

            let instruction = format!("{:?}", cm.tensor_graph.operations[*op_id]);

            println!(
                "  Operation {} (Layer {:?} - {})",
                op_id, layer_id, layer_name
            );
            println!("  Instruction: {}", instruction);

            let inputs = cm.tensor_graph.get_operation_inputs(*op_id);
            println!("  Inputs:");
            for input in inputs {
                let tensor = cm.tensor_read(input);
                let shape = format!("{:?}", tensor.desc.to_dims());

                let location = match &tensor.device {
                    DeviceId::CPU => "CPU".to_string(),
                    DeviceId::GPU(gpu_idx) => format!("GPU {}", gpu_idx),
                };

                let producers: String = cm
                    .tensor_graph
                    .get_tensor_producers(input)
                    .iter()
                    .map(|&op| format!("{}", op))
                    .collect::<Vec<_>>()
                    .join(", ");

                println!(
                    "    Tensor {} - Shape: {} - Location: {} - Producers: {}",
                    input,
                    shape,
                    location,
                    if producers.is_empty() {
                        "None".to_string()
                    } else {
                        producers
                    }
                );
            }

            let outputs = cm.tensor_graph.get_operation_outputs(*op_id);
            println!("  Outputs:");
            for output in outputs {
                let tensor = cm.tensor_read(output);
                let shape = format!("{:?}", tensor.desc.to_dims());

                let location = match &tensor.device {
                    DeviceId::CPU => "CPU".to_string(),
                    DeviceId::GPU(gpu_idx) => format!("GPU {}", gpu_idx),
                };

                let consumers: Vec<String> = cm
                    .tensor_graph
                    .get_tensor_consumers(output)
                    .iter()
                    .map(|&op| format!("{}", op))
                    .collect();

                println!(
                    "    Tensor {} - Shape: {} - Location: {} - Consumers: {}",
                    output,
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

    println!("\n=== ORIGINAL MODEL LAYER CONNECTIONS ===\n");

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
                    tensor.desc.to_dims(),
                    cm.format_memory_mb(tensor.desc.size_in_bytes() as u64)
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
    println!("Execution Stages: {}", execution_plan.len());

    let total_memory = cm.tensor_graph.memory_requirements;
    println!(
        "\nMemory Requirements: {}",
        cm.format_memory_mb(total_memory as u64)
    );
}
