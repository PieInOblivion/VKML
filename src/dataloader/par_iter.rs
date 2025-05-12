use std::collections::VecDeque;
use std::sync::Arc;

use crate::thread_pool::worker::WorkFuture;

use super::{
    data_batch::DataBatch,
    dataloader::{DataLoader, DatasetSplit},
};

pub struct MultithreadedIterator<T: DataLoader> {
    dataloader: Arc<T>,
    split: DatasetSplit,
    next_batch: usize,
    pending_futures: VecDeque<WorkFuture>,
    max_pending: usize,
}

impl<T: DataLoader> MultithreadedIterator<T> {
    fn new(dl: T, split: DatasetSplit) -> Self {
        let max_pending = dl.get_config().prefetch_count;
        let mut iterator = MultithreadedIterator {
            dataloader: Arc::new(dl),
            split,
            next_batch: 0,
            pending_futures: VecDeque::with_capacity(max_pending),
            max_pending,
        };

        iterator.request_next_batches();

        iterator
    }

    fn request_next_batches(&mut self) {
        while self.pending_futures.len() < self.max_pending {
            let batch_number = self.next_batch + self.pending_futures.len();

            if let Some(batch_data) = self
                .dataloader
                .get_batch_reference(self.split, batch_number)
            {
                let work = self.dataloader.create_batch_work(batch_number, batch_data);
                let future = self.dataloader.get_thread_pool().submit_work(work);
                self.pending_futures.push_back(future);
            } else {
                break;
            }
        }
    }

    fn wait_for_next_batch(&mut self) -> Option<DataBatch> {
        let future = self.pending_futures.pop_front()?;
        Some(
            self.dataloader
                .process_work_result(future.wait_and_take(), self.next_batch),
        )
    }
}

impl<T: DataLoader> Iterator for MultithreadedIterator<T> {
    type Item = DataBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(batch) = self.wait_for_next_batch() {
            self.next_batch += 1;
            self.request_next_batches();
            Some(batch)
        } else {
            None
        }
    }
}

pub trait MultithreadedDataLoaderIterator: DataLoader {
    fn par_iter(self, split: DatasetSplit) -> MultithreadedIterator<Self>
    where
        Self: Sized;
}

impl<T: DataLoader> MultithreadedDataLoaderIterator for T {
    fn par_iter(self, split: DatasetSplit) -> MultithreadedIterator<T> {
        MultithreadedIterator::new(self, split)
    }
}
