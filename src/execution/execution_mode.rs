#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    Inference,
    Training,    // Forward pass + backward pass + optimiser updates
    Calibration, // Forward pass + quantisation statistics collection
    Pruning,     // Forward + backward + importance scoring for model compression
    Debugging,   // Forward pass + debug outputs for development
}

impl ExecutionMode {
    pub fn all() -> Vec<ExecutionMode> {
        vec![
            ExecutionMode::Inference,
            ExecutionMode::Training,
            ExecutionMode::Calibration,
            ExecutionMode::Pruning,
            ExecutionMode::Debugging,
        ]
    }
}
