use std::sync::Arc;

use crate::thread_pool::{
    thread_pool::ThreadPool,
    worker::{WorkResult, WorkType},
};

use super::{config::DataLoaderConfig, data_batch::DataBatch, error::VKMLEngineError};

#[derive(Copy, Clone)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}

#[derive(Clone, Copy, Debug)]
pub enum SourceFormat {
    U8,
    U16,
    F32,
}

impl SourceFormat {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            SourceFormat::U8 => 1,
            SourceFormat::U16 => 2,
            SourceFormat::F32 => 4,
        }
    }
}

pub trait DataLoader {
    type BatchDataReference;

    fn get_batch_reference(
        &self,
        split: DatasetSplit,
        batch_number: usize,
    ) -> Option<Self::BatchDataReference>;
    fn shuffle_whole_dataset(&mut self) -> Result<(), VKMLEngineError>;
    fn shuffle_individual_datasets(&mut self) -> Result<(), VKMLEngineError>;
    fn len(&self) -> usize;
    fn get_config(&self) -> &DataLoaderConfig;
    fn get_thread_pool(&self) -> Arc<ThreadPool>;

    fn create_batch_work(
        &self,
        batch_number: usize,
        batch_data_ref: Self::BatchDataReference,
    ) -> WorkType;
    fn process_work_result(&self, result: WorkResult, expected_batch: usize) -> DataBatch;
}
