use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::thread;

use rand::rngs::StdRng;

use crate::thread_pool::thread_pool::ThreadPool;
use crate::utils::dataloader_error::DataLoaderError;

// TODO: Make bad values impossible using NonZeroUsize etc
// Downside is it becomes annoying to use having to .into() and NonZeroUsize::new everywhere...
// Mayber just assert!() instead?
// TODO: Clean up names
pub struct DataLoaderConfig {
    pub data_loading_threads: usize,
    pub prefetch_threads: usize,
    pub prefetch_count: usize,
    pub batch_size: usize,
    pub train_ratio: f32,
    pub test_ratio: f32,
    pub sort_dataset: bool,
    pub shuffle: bool,
    pub shuffle_seed: Option<u64>,
    pub rng: Option<Arc<Mutex<StdRng>>>,
    pub drop_last: bool,
    pub thread_pool: ThreadPool
}

impl DataLoaderConfig {
    pub fn build(self) -> Result<Self, DataLoaderError> {
        check_split_ratios(self.train_ratio, self.test_ratio)?;

        Ok(self)
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism().map(NonZero::get).unwrap_or(1);
        
        Self {
            data_loading_threads: num_cpus,
            prefetch_threads: 4,
            prefetch_count: 4,
            batch_size: 32,
            train_ratio: 0.8,
            test_ratio: 0.1,
            sort_dataset: false,
            shuffle: true,
            shuffle_seed: None,
            rng: None,
            drop_last: true,
            thread_pool: ThreadPool::new(num_cpus)
        }
    }
}

fn check_split_ratios(train_ratio: f32, test_ratio: f32) -> Result<(), DataLoaderError> {
    if train_ratio + test_ratio > 1.0 || train_ratio <= 0.0 || test_ratio < 0.0 {
        return Err(DataLoaderError::InvalidSplitRatios {
            train: train_ratio,
            test: test_ratio,
        });
    }
    Ok(())
}
