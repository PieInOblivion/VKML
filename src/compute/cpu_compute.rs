use super::memory_tracker::MemoryTracker;

pub struct CPUCompute {
    pub memory_tracking: MemoryTracker
}

// TODO: Implement CPU layer computations, forward, backward etc.
// TODO: Memory calculations don't update which means it wont account this applications usage

impl CPUCompute {
    pub fn new(memory_limit_bytes: Option<u64>) -> Self {
        // This implementation will by default use only available memory
        // This does not include swap capacity etc
        // This allows for an override if user knows better
        if let Some(limit) = memory_limit_bytes {
            Self {
                memory_tracking: MemoryTracker::new(limit)
            }
        } else {
            Self {
                // If cannot get available system memory, assume there is none
                // Crate returns in kilobytes. We need bytes
                // TODO: Print warning to user if sys_info returns 0 or error
                memory_tracking: MemoryTracker::new(sys_info::mem_info().map(|info| info.avail * 1024).unwrap_or(0))
            }
        }
    }
}