use crate::{
    compute::compute_manager::ComputeManager,
    dataloader::error::VKMLError,
    model::layer_connection::{LayerConnection, LayerId},
    tensor::storage::TensorStorage,
    tensor::storage::TensorStorageOps,
    tensor_graph::tensor_graph::TensorId,
};

pub fn print_model_stats(cm: &ComputeManager) {
    let mut total_memory = 0u64;

    println!("\nModel Statistics");
    println!("================");
    println!("\nBatch Size: {}", cm.model.batch_size);
    println!("\nLayer Details:");
    println!("{:-<125}", "");
    println!(
        "{:<4} {:<12} {:<10} {:<18} {:<18} {:<12} {:<20} {}",
        "ID", "Type", "Memory", "Input Shape", "Output Shape", "Device", "Connections", "Config"
    );
    println!("{:-<125}", "");

    let execution_order = match &cm.model.verified {
        Some(verified) => &verified.execution_order,
        None => {
            println!("Warning: Model not verified, execution order may be incorrect");
            return;
        }
    };

    let mut ordered_layer_ids: Vec<LayerId> = execution_order.to_vec();
    ordered_layer_ids.sort();

    for &layer_id in &ordered_layer_ids {
        if let Some(layer) = cm.model.layers.get(&layer_id) {
            let layer_tensor_ids: Vec<TensorId> = (0..cm.tensor_graph.tensors.len())
                .filter(|&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(layer_id)))
                .collect();

            let is_output_layer = layer.output_connections.is_empty();

            let layer_output_tensors: Vec<TensorId> = if is_output_layer {
                cm.tensor_graph
                    .output_tensors
                    .iter()
                    .filter(|&&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(layer_id)))
                    .cloned()
                    .collect()
            } else {
                // For non-output layers, find tensors that other layers consume
                layer_tensor_ids
                    .iter()
                    .filter(|&&tensor_id| {
                        cm.tensor_graph
                            .get_tensor_consumers(tensor_id)
                            .iter()
                            .any(|&op_id| {
                                let op_inputs = cm.tensor_graph.get_operation_inputs(op_id);
                                let first_input = op_inputs.first().cloned();
                                let op_layer = first_input.and_then(|id| {
                                    cm.tensor_graph.tensor_to_layer.get(id).cloned().flatten()
                                });
                                op_layer.is_some() && op_layer != Some(layer_id)
                            })
                    })
                    .cloned()
                    .collect()
            };

            // Get representative output tensor for shape info
            let output_tensor = if !layer_output_tensors.is_empty() {
                layer_output_tensors[0]
            } else if !layer_tensor_ids.is_empty() {
                // No explicit outputs, use any tensor (preferably one produced by an operation)
                *layer_tensor_ids
                    .iter()
                    .find(|&&id| !cm.tensor_graph.get_tensor_producers(id).is_empty())
                    .unwrap_or(&layer_tensor_ids[0])
            } else {
                // No tensors found at all - should never happen
                continue;
            };

            let input_shapes_str = if layer.input_connections.is_empty() {
                "None".to_string()
            } else {
                layer
                    .input_connections
                    .iter()
                    .filter_map(|conn| {
                        let source_layer_id = conn.get_layerid();
                        let output_idx = conn.get_outputidx();

                        let connected_layer_outputs = get_layer_output_tensors(cm, source_layer_id);

                        if output_idx < connected_layer_outputs.len() {
                            let output_tensor_id = connected_layer_outputs[output_idx];
                            if let Some(tensor) = cm.tensor_graph.tensors.get(output_tensor_id) {
                                let dims = tensor.desc.to_dims();
                                return Some(format_dimensions(&dims));
                            }
                        }
                        None
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            let output_shapes_str = if let Some(tensor) = cm.tensor_graph.tensors.get(output_tensor)
            {
                format_dimensions(&tensor.desc.to_dims())
            } else {
                "Unknown".to_string()
            };

            let connections_str =
                format_layer_connections(&layer.input_connections, &layer.output_connections);

            let memory_bytes = layer_tensor_ids
                .iter()
                .filter_map(|&id| cm.tensor_graph.tensors.get(id))
                .map(|t| t.desc.size_in_bytes() as u64)
                .sum();

            let device_location = match &cm.tensor_graph.tensors[output_tensor].data {
                TensorStorage::CPU(_) => "CPU".to_string(),
                TensorStorage::GPU(gpu_storage) => {
                    format!("GPU {}", gpu_storage.gpu_idx().unwrap())
                }
                TensorStorage::Unallocated(_) => "Unallocated".to_string(),
            };

            let layer_type = layer.layer.name();
            let layer_config = layer.layer.config_string().unwrap_or_default();

            println!(
                "{:<4} {:<12} {:<10} {:<18} {:<18} {:<12} {:<20} {}",
                layer_id,
                layer_type,
                cm.format_memory_mb(memory_bytes),
                input_shapes_str,
                output_shapes_str,
                device_location,
                connections_str,
                layer_config
            );

            total_memory += memory_bytes;
        }
    }

    println!("{:-<125}", "");

    let mut entry_points = cm
        .tensor_graph
        .input_tensors
        .iter()
        .filter_map(|&id| cm.tensor_graph.tensor_to_layer.get(id).cloned().flatten())
        .collect::<Vec<_>>();

    let mut exit_points = cm
        .tensor_graph
        .output_tensors
        .iter()
        .filter_map(|&id| cm.tensor_graph.tensor_to_layer.get(id).cloned().flatten())
        .collect::<Vec<_>>();

    entry_points.sort();
    exit_points.sort();
    entry_points.dedup();
    exit_points.dedup();

    println!("\nGraph Structure:");
    println!("Entry points: {:?}", entry_points);
    println!("Exit points: {:?}", exit_points);

    println!("\nModel Summary:");
    println!("Total Memory: {}", cm.format_memory_mb(total_memory));

    println!("\nMemory Allocation:");
    for (device, used, available) in cm.get_memory_usage_summary() {
        println!("{} Memory Used: {}", device, used);
        println!("{} Memory Available: {}", device, available);
    }
}

fn format_dimensions(dims: &[usize]) -> String {
    if dims.len() <= 4 {
        dims.iter()
            .map(|&d| d.to_string())
            .collect::<Vec<_>>()
            .join("×")
    } else {
        format!("{}d tensor", dims.len())
    }
}

fn get_layer_output_tensors(cm: &ComputeManager, layer_id: LayerId) -> Vec<TensorId> {
    // Get all tensors belonging to this layer
    let layer_tensors: Vec<TensorId> = (0..cm.tensor_graph.tensors.len())
        .filter(|&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(layer_id)))
        .collect();

    // First check explicit output tensors
    let explicit_outputs: Vec<TensorId> = cm
        .tensor_graph
        .output_tensors
        .iter()
        .filter(|&&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(layer_id)))
        .cloned()
        .collect();

    if !explicit_outputs.is_empty() {
        return explicit_outputs;
    }

    let mut outputs = Vec::new();

    if let Some(layer) = cm.model.layers.get(&layer_id) {
        if layer.output_connections.is_empty() {
            // This is a final layer, so all its tensors could be outputs
            outputs.extend(layer_tensors.clone());
        } else {
            for &tensor_id in &layer_tensors {
                let consumers = cm.tensor_graph.get_tensor_consumers(tensor_id);
                let is_consumed_by_other_layer = consumers.iter().any(|&op_id| {
                    let op_inputs = cm.tensor_graph.get_operation_inputs(op_id);
                    let first_input = op_inputs.first().cloned();
                    let op_layer = first_input
                        .and_then(|id| cm.tensor_graph.tensor_to_layer.get(id).cloned().flatten());
                    op_layer.is_some() && op_layer != Some(layer_id)
                });

                if is_consumed_by_other_layer {
                    outputs.push(tensor_id);
                }
            }
        }
    }

    // If we still haven't found outputs, include all tensors that have producers
    if outputs.is_empty() {
        for &tensor_id in &layer_tensors {
            if !cm.tensor_graph.get_tensor_producers(tensor_id).is_empty() {
                outputs.push(tensor_id);
            }
        }
    }

    outputs
}

pub fn print_layer_values(cm: &ComputeManager, layer_id: LayerId) -> Result<(), VKMLError> {
    let layer = cm
        .model
        .layers
        .get(&layer_id)
        .ok_or(VKMLError::VulkanLoadError(format!(
            "Layer ID {} not found",
            layer_id
        )))?;

    println!("\nLayer {} Values ({})", layer_id, layer.layer.name());
    println!("{:-<120}", "");
    println!("Input connections: {:?}", layer.input_connections);
    println!("Output connections: {:?}", layer.output_connections);

    let format_array = |arr: &[f32], max_items: usize| {
        let mut s = String::from("[");
        for (i, val) in arr.iter().take(max_items).enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{:.6}", val));
        }
        if arr.len() > max_items {
            s.push_str(", ...")
        }
        s.push(']');
        s
    };

    let print_tensor_stats = |data: &[f32]| {
        if !data.is_empty() {
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let variance =
                data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            let std_dev = variance.sqrt();

            println!("  Stats:");
            println!("    Min: {:.6}", min_val);
            println!("    Max: {:.6}", max_val);
            println!("    Mean: {:.6}", mean);
            println!("    Std Dev: {:.6}", std_dev);

            let non_zero = data.iter().filter(|&&x| x != 0.0).count();
            println!(
                "    Non-zero elements: {} ({:.2}%)",
                non_zero,
                (non_zero as f32 / data.len() as f32) * 100.0
            );
        }
    };

    // Get all tensors belonging to this layer
    let tensor_ids: Vec<TensorId> = (0..cm.tensor_graph.tensors.len())
        .filter(|&id| cm.tensor_graph.tensor_to_layer.get(id) == Some(&Some(layer_id)))
        .collect();

    for tensor_id in &tensor_ids {
        let tensor = &cm.tensor_graph.tensors[*tensor_id];
        let data = tensor.data.get_data();
        let gpu_idx = tensor.data.gpu_idx();

        println!("\nTensor {}:", tensor_id);
        println!(
            "  Location: {}",
            match gpu_idx {
                Some(idx) => format!("GPU {}", idx),
                None => "CPU".to_string(),
            }
        );
        println!("  Shape: {:?}", tensor.desc.to_dims());
        println!(
            "  Size in memory: {}",
            cm.format_memory_mb((data.len() * std::mem::size_of::<f32>()) as u64)
        );
        println!("  Values: {}", format_array(&data, 10));

        print_tensor_stats(&data);
    }

    println!("\nLayer Output Tensors:");

    let output_tensors: Vec<_> = tensor_ids
        .iter()
        .filter(|&&id| {
            // Output tensor is either:
            // 1. Explicitly marked as model output
            cm.tensor_graph.output_tensors.contains(&id) ||
            // 2. Used as input by operations in other layers
            cm.tensor_graph.get_tensor_consumers(id).iter().any(|&op_id| {
                let op_inputs = cm.tensor_graph.get_operation_inputs(op_id);
                let first_input = op_inputs.first().cloned();
                let op_layer = first_input.and_then(|id| cm.tensor_graph.tensor_to_layer.get(id).cloned().flatten());
                op_layer.is_some() && op_layer != Some(layer_id)
            })
        })
        .collect();

    if output_tensors.is_empty() {
        println!("  No explicit output tensors found");
    } else {
        for &tensor_id in output_tensors {
            let tensor = &cm.tensor_graph.tensors[tensor_id];
            println!("  Tensor {} Shape: {:?}", tensor_id, tensor.desc.to_dims());
        }
    }

    Ok(())
}

fn format_layer_connections(inputs: &[LayerConnection], outputs: &[LayerConnection]) -> String {
    // Maintain original order of connections
    let in_ids: Vec<String> = inputs
        .iter()
        .map(|conn| match conn {
            LayerConnection::DefaultOutput(id) => format!("{}:0", id),
            LayerConnection::SpecificOutput(id, idx) => format!("{}:{}", id, idx),
        })
        .collect();

    let out_ids: Vec<String> = outputs
        .iter()
        .map(|conn| match conn {
            LayerConnection::DefaultOutput(id) => format!("{}:0", id),
            LayerConnection::SpecificOutput(id, idx) => format!("{}:{}", id, idx),
        })
        .collect();

    if in_ids.is_empty() && out_ids.is_empty() {
        return "None".to_string();
    }

    let mut result = String::new();

    if !in_ids.is_empty() {
        result.push_str(&format!("in:[{}]", in_ids.join(",")));
    }

    if !out_ids.is_empty() {
        if !result.is_empty() {
            result.push_str(" ");
        }
        result.push_str(&format!("out:[{}]", out_ids.join(",")));
    }

    result
}
