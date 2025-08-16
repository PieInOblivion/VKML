use super::{data_batch::DataBatch, dataloader::DataLoader};
use std::collections::VecDeque;
use std::sync::Arc;
use zero_pool::{zp_define_task_fn, zp_task_params};

/// A pending batch with its future and the allocated data
struct PendingBatch {
    batch: DataBatch,
    future: zero_pool::WorkFuture,
}

/// Extension trait for parallel iteration using zero-pool
pub trait DataLoaderParIter: DataLoader {
    /// Create a parallel iterator over a specific split
    fn par_iter(self, split_idx: usize) -> ParallelDataIterator<Self>
    where
        Self: Sized,
    {
        ParallelDataIterator::new(self, split_idx)
    }
}

impl<T: DataLoader> DataLoaderParIter for T {}

/// High-performance parallel iterator with prefetching using zero-pool
pub struct ParallelDataIterator<T: DataLoader> {
    dataloader: Arc<T>,
    split_idx: usize,
    current_batch: usize,
    total_batches: usize,
    pending_batches: VecDeque<PendingBatch>,
    max_pending: usize,
}

impl<T: DataLoader> ParallelDataIterator<T> {
    fn new(dl: T, split_idx: usize) -> Self {
        let total_batches = dl.batches_in_split(split_idx);
        let max_pending = dl.get_config().prefetch_count;

        let mut iterator = Self {
            dataloader: Arc::new(dl),
            split_idx,
            current_batch: 0,
            total_batches,
            pending_batches: VecDeque::with_capacity(max_pending),
            max_pending,
        };

        // Start prefetching immediately
        iterator.request_next_batches();
        iterator
    }

    /// Keep the prefetch pipeline full by submitting new batch loading tasks
    fn request_next_batches(&mut self) {
        while self.pending_batches.len() < self.max_pending {
            let batch_number = self.current_batch + self.pending_batches.len();

            if batch_number >= self.total_batches {
                break;
            }

            if let Some(indices) = self
                .dataloader
                .get_batch_indices(self.split_idx, batch_number)
            {
                let pending_batch = self.submit_batch_loading_task(&indices);
                self.pending_batches.push_back(pending_batch);
            } else {
                break;
            }
        }
    }

    /// Submit a batch loading task to zero-pool and return the pending batch
    fn submit_batch_loading_task(&self, indices: &[usize]) -> PendingBatch {
        let bytes_per_item = self.dataloader.bytes_per_item();
        let data_type = self.dataloader.batch_data_type();
        let total_bytes = indices.len() * bytes_per_item;
        let mut batch = DataBatch::new(total_bytes, data_type);

        // Create loading tasks for each item
        let tasks: Vec<_> = indices
            .iter()
            .enumerate()
            .map(|(i, &index)| {
                LoadItemTask::new(
                    Arc::as_ptr(&self.dataloader) as *const dyn DataLoader,
                    index,
                    batch.as_mut_ptr(),
                    i * bytes_per_item,
                )
            })
            .collect();

        // Submit batch to zero-pool
        let future = self
            .dataloader
            .get_thread_pool()
            .submit_batch_uniform(load_item_task, &tasks);

        PendingBatch { batch, future }
    }

    /// Wait for the next batch to complete and return it
    fn wait_for_next_batch(&mut self) -> Option<DataBatch> {
        let pending = self.pending_batches.pop_front()?;
        pending.future.wait();
        Some(pending.batch)
    }
}

impl<T: DataLoader> Iterator for ParallelDataIterator<T> {
    type Item = DataBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.total_batches {
            return None;
        }

        let batch = self.wait_for_next_batch()?;
        self.current_batch += 1;
        self.request_next_batches();

        Some(batch)
    }
}

// Zero-pool task for loading individual items
zp_task_params! {
    LoadItemTask {
        dataloader: *const dyn DataLoader,
        index: usize,
        dest_ptr: *mut u8,
        offset: usize,
    }
}

zp_define_task_fn!(load_item_task, LoadItemTask, |params| {
    let item_data = unsafe { (*params.dataloader).get_item_bytes(params.index) };
    unsafe {
        std::ptr::copy_nonoverlapping(
            item_data.as_ptr(),
            params.dest_ptr.add(params.offset),
            item_data.len(),
        );
    }
});
