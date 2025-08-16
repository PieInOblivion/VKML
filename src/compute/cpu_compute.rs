use std::sync::Arc;

use zero_pool::ThreadPool;

use super::memory_tracker::MemoryTracker;

pub struct CPUCompute {
    pub memory_tracking: MemoryTracker,
    thread_pool: Arc<ThreadPool>,
}

// TODO: Implement CPU layer computations, forward, backward etc.
// TODO: Memory calculations don't update which means it wont account this applications usage

impl CPUCompute {
    pub fn new(memory_limit_bytes: Option<u64>, thread_pool: Arc<ThreadPool>) -> Self {
        // This implementation will by default use only available memory
        // This does not include swap capacity etc
        // This allows for an override if user knows better

        let memory_limit = memory_limit_bytes.unwrap_or_else(|| {
            // If cannot get available system memory, assume there is none
            // Crate returns in kilobytes. We need bytes
            // TODO: Print warning to user if sys_info returns 0 or error
            sys_info::mem_info()
                .map(|info| info.avail * 1024)
                .unwrap_or(0)
        });

        Self {
            memory_tracking: MemoryTracker::new(memory_limit),
            thread_pool: thread_pool.clone(),
        }
    }
}
