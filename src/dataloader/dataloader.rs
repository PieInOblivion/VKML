use crate::dataloader::{config::DataLoaderConfig, data_type::DataType};

use super::error::VKMLError;

pub trait DataLoader {
    /// Returns (total_dataset_size, split_lengths)
    /// split_lengths[i] contains the number of items in split i
    fn len(&self) -> (usize, Vec<usize>);

    /// Get the indices for a specific batch in a split
    /// Returns None if the batch doesn't exist (beyond dataset bounds)
    fn get_batch_indices(&self, split_idx: usize, batch_idx: usize) -> Option<Vec<usize>>;

    /// Number of bytes required to store one item
    fn bytes_per_item(&self) -> usize;

    /// The format of data stored in a batch
    fn batch_data_type(&self) -> DataType;

    /// Number of bytes required for a full batch
    /// Default implementation: batch_size * bytes_per_item
    fn bytes_per_batch(&self) -> usize {
        self.get_batch_size() * self.bytes_per_item()
    }

    /// Get the configured batch size
    fn get_batch_size(&self) -> usize;

    /// Get the bytes for a single item
    fn get_item_bytes(&self, index: usize) -> Vec<u8>;

    /// Shuffle the entire dataset (all splits together)
    fn shuffle_whole_dataset(&mut self) -> Result<(), VKMLError>;

    /// Shuffle each split individually (preserves split boundaries)
    fn shuffle_individual_dataset(&mut self, split_idx: usize) -> Result<(), VKMLError>;

    /// Get the number of batches in a specific split
    fn batches_in_split(&self, split_idx: usize) -> usize {
        let (_, split_lengths) = self.len();
        if split_idx >= split_lengths.len() {
            return 0;
        }

        let split_size = split_lengths[split_idx];
        let batch_size = self.get_batch_size();

        if self.get_config().drop_last {
            split_size / batch_size
        } else {
            (split_size + batch_size - 1) / batch_size
        }
    }

    /// Get the total number of splits
    fn num_splits(&self) -> usize {
        self.len().1.len()
    }

    /// Get access to the dataloader configuration
    fn get_config(&self) -> &DataLoaderConfig;

    fn get_thread_pool(&self) -> &zero_pool::ThreadPool;
}
