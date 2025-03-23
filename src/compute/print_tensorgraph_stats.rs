use crate::{
    compute::compute_manager::ComputeManager,
    tensor::tensor_data::TensorData,
    tensor_graph::tensor_graph::{OperationId, TensorId},
};
use std::collections::HashSet;

pub fn print_tensor_flow(cm: &ComputeManager) {
    println!("\n=== TENSOR GRAPH VISUALIZATION ===\n");

    let execution_plan = cm.tensor_graph.create_execution_plan();

    println!("Execution Plan: {} stages", execution_plan.len());
    println!("{:-<100}", "");

    let mut produced_tensors = HashSet::new();

    produced_tensors.extend(cm.tensor_graph.input_tensors.iter().cloned());

    // Add parameter tensors (tensors with no dependencies that aren't inputs)
    for (tensor_id, _) in &cm.tensor_graph.tensors {
        if !cm.tensor_graph.input_tensors.contains(tensor_id)
            && (!cm.tensor_graph.tensor_dependencies.contains_key(tensor_id)
                || cm.tensor_graph.tensor_dependencies[tensor_id].is_empty())
        {
            produced_tensors.insert(tensor_id.clone());
        }
    }

    for (stage_idx, operations) in execution_plan.iter().enumerate() {
        println!("\nStage {}: {} operations", stage_idx + 1, operations.len());
        println!("{:-<100}", "");

        for op_id in operations {
            let layer_id = op_id.0;
            let instruction_idx = op_id.1;

            let layer_name = cm
                .model
                .layers
                .get(&layer_id)
                .map(|layer| layer.layer.name())
                .unwrap_or_else(|| "Unknown".to_string());

            let instruction = cm
                .tensor_graph
                .operations
                .get(op_id)
                .map(|i| format!("{:?}", i))
                .unwrap_or_else(|| "Unknown".to_string());

            println!(
                "  Operation {:?} (Layer {} - {})",
                op_id, layer_id, layer_name
            );
            println!("  Instruction: {}", instruction);

            if let Some(inputs) = cm.tensor_graph.operation_inputs.get(op_id) {
                println!("  Inputs:");
                for input in inputs {
                    let tensor = cm.tensor_graph.tensors.get(input);
                    let shape = tensor
                        .map(|t| format!("{:?}", t.desc.to_dims()))
                        .unwrap_or_else(|| "Unknown".to_string());

                    let status = if produced_tensors.contains(input) {
                        "✓ AVAILABLE"
                    } else {
                        "✗ NOT AVAILABLE"
                    };

                    let location = tensor
                        .map(|t| match &t.data {
                            TensorData::CPU(_) => "CPU".to_string(),
                            TensorData::GPU { gpu_idx, .. } => format!("GPU {}", gpu_idx),
                            TensorData::Unallocated => "Unallocated".to_string(),
                        })
                        .unwrap_or("Unknown".to_string());

                    // Determine tensor role structurally
                    let role = if cm.tensor_graph.input_tensors.contains(input) {
                        "INPUT"
                    } else if cm
                        .tensor_graph
                        .tensor_dependencies
                        .get(input)
                        .map_or(true, |deps| deps.is_empty())
                    {
                        "PARAMETER"
                    } else {
                        "INTERMEDIATE"
                    };

                    let producers = cm
                        .tensor_graph
                        .tensor_dependencies
                        .get(input)
                        .map(|deps| {
                            deps.iter()
                                .map(|op| format!("{:?}", op))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_else(|| "None".to_string());

                    // Display tensor name in quotes to indicate it's arbitrary
                    println!(
                        "    TensorId({}, \"{}\") - [{}] - Shape: {} - {} - Location: {} - Producers: {}",
                        input.0, input.1, role, shape, status, location, producers
                    );
                }
            }

            if let Some(outputs) = cm.tensor_graph.operation_outputs.get(op_id) {
                println!("  Outputs:");
                for output in outputs {
                    let tensor = cm.tensor_graph.tensors.get(output);
                    let shape = tensor
                        .map(|t| format!("{:?}", t.desc.to_dims()))
                        .unwrap_or_else(|| "Unknown".to_string());

                    let location = tensor
                        .map(|t| match &t.data {
                            TensorData::CPU(_) => "CPU".to_string(),
                            TensorData::GPU { gpu_idx, .. } => format!("GPU {}", gpu_idx),
                            TensorData::Unallocated => "Unallocated".to_string(),
                        })
                        .unwrap_or("Unknown".to_string());

                    // Determine tensor role structurally
                    let role = if cm.tensor_graph.output_tensors.contains(output) {
                        "OUTPUT"
                    } else {
                        "INTERMEDIATE"
                    };

                    let consumers =
                        find_tensor_consumers(output, &cm.tensor_graph.operation_inputs);

                    // Display tensor name in quotes to indicate it's arbitrary
                    println!(
                        "    TensorId({}, \"{}\") - [{}] - Shape: {} - Location: {} - Consumers: {}",
                        output.0, output.1, role, shape, location, consumers
                    );

                    produced_tensors.insert(output.clone());
                }
            }

            println!();
        }
    }

    println!("\n=== ORIGINAL MODEL LAYER CONNECTIONS ===\n");

    // Sort layer IDs for consistent output
    let mut layer_ids: Vec<_> = cm.model.layers.keys().cloned().collect();
    layer_ids.sort();

    for layer_id in &layer_ids {
        if let Some(layer) = cm.model.layers.get(&layer_id) {
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
            let layer_tensors: Vec<_> = cm
                .tensor_graph
                .tensors
                .iter()
                .filter(|(id, _)| id.0 == *layer_id)
                .collect();

            for (tensor_id, tensor) in layer_tensors {
                // Determine tensor role structurally
                let role = if cm.tensor_graph.input_tensors.contains(tensor_id) {
                    "INPUT"
                } else if cm.tensor_graph.output_tensors.contains(tensor_id) {
                    "OUTPUT"
                } else if cm
                    .tensor_graph
                    .tensor_dependencies
                    .get(tensor_id)
                    .map_or(true, |deps| deps.is_empty())
                {
                    "PARAMETER"
                } else {
                    "INTERMEDIATE"
                };

                // Display tensor name in quotes to indicate it's arbitrary
                println!(
                    "    \"{}\": [{}] - Shape {:?}, Size: {}",
                    tensor_id.1,
                    role,
                    tensor.desc.to_dims(),
                    cm.format_memory_mb(tensor.desc.size_in_bytes() as u64)
                );
            }

            println!();
        }
    }

    println!("\n=== TENSOR GRAPH SUMMARY ===\n");
    println!("Total Tensors: {}", cm.tensor_graph.tensors.len());
    println!("Total Operations: {}", cm.tensor_graph.operations.len());
    println!("Input Tensors: {}", cm.tensor_graph.input_tensors.len());
    println!("Output Tensors: {}", cm.tensor_graph.output_tensors.len());
    println!("Execution Stages: {}", execution_plan.len());

    let total_memory = cm.tensor_graph.calculate_memory_requirements();
    println!(
        "\nMemory Requirements: {}",
        cm.format_memory_mb(total_memory)
    );

    let mut total_params = 0;
    for layer_id in &layer_ids {
        total_params += cm.calculate_layer_parameters(*layer_id);
    }
    println!("Total Parameters: {}", total_params);
}

fn find_tensor_consumers(
    tensor_id: &TensorId,
    operation_inputs: &std::collections::HashMap<OperationId, HashSet<TensorId>>,
) -> String {
    let consumers = operation_inputs
        .iter()
        .filter(|(_, inputs)| inputs.contains(tensor_id))
        .map(|(op_id, _)| format!("{:?}", op_id))
        .collect::<Vec<_>>();

    if consumers.is_empty() {
        "None".to_string()
    } else {
        consumers.join(", ")
    }
}
