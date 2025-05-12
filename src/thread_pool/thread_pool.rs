use std::{
    collections::VecDeque,
    num::NonZero,
    sync::{Arc, Condvar, Mutex, atomic::AtomicUsize},
};

use super::worker::{WorkFuture, WorkFutureBatch, WorkItem, WorkQueue, WorkType, Worker};

pub struct ThreadPool {
    work_queue: Arc<WorkQueue>,
    workers: Vec<Worker>,
}

impl ThreadPool {
    pub fn new() -> Arc<ThreadPool> {
        let num_threads = std::thread::available_parallelism()
            .map(NonZero::get)
            .unwrap_or(1);

        Self::new_with(num_threads)
    }

    pub fn new_with(num_threads: usize) -> Arc<ThreadPool> {
        // Simplier than using NonZeroUsize and .get()
        assert!(num_threads > 0);

        let work_queue = Arc::new(WorkQueue {
            queue: Mutex::new(VecDeque::new()),
            items_count: AtomicUsize::new(0),
            condvar: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(num_threads);
        for id in 0..num_threads {
            workers.push(Worker::new(id, Arc::clone(&work_queue)));
        }

        Arc::new(ThreadPool {
            work_queue,
            workers,
        })
    }

    pub fn submit_work(&self, work: WorkType) -> WorkFuture {
        let future = WorkFuture::new();

        let work_item = WorkItem {
            work,
            future: future.clone(),
        };

        self.work_queue.submit_work_item(work_item);

        self.work_queue.condvar.notify_one();

        future
    }

    pub fn submit_batch(&self, work_items: Vec<WorkType>) -> WorkFutureBatch {
        let mut futures = Vec::with_capacity(work_items.len());
        let mut batch_items = Vec::with_capacity(work_items.len());

        for work in work_items {
            let future = WorkFuture::new();
            futures.push(future.clone());

            batch_items.push(WorkItem { work, future });
        }

        self.work_queue.submit_work_batch(batch_items);

        for _ in 0..futures.len().min(self.workers.len()) {
            self.work_queue.condvar.notify_one();
        }

        WorkFutureBatch { futures }
    }
}
