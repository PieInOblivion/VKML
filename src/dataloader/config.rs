use std::sync::{Arc, Mutex};

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::error::VKMLError;

pub struct DataLoaderConfig {
    pub prefetch_count: usize,
    pub batch_size: usize,
    pub split_ratios: Vec<f32>,
    pub sort_dataset: bool,
    pub shuffle_seed: Option<u64>,
    pub rng: Option<Arc<Mutex<StdRng>>>,
    pub drop_last: bool,
}

impl DataLoaderConfig {
    pub fn build(mut self) -> Self {
        assert!(
            !self.split_ratios.is_empty(),
            "Split ratios cannot be empty"
        );

        let sum: f32 = self.split_ratios.iter().sum();
        assert!(
            (sum - 1.0).abs() <= 1e-6,
            "Split ratios must sum to 1.0, got {}",
            sum
        );

        assert!(
            self.split_ratios.iter().all(|&ratio| ratio > 0.0),
            "All split ratios must be positive, got: {:?}",
            self.split_ratios
        );

        if self.shuffle_seed.is_none() {
            self.shuffle_seed = Some(rand::rng().random());
        }

        self.rng = Some(Arc::new(Mutex::new(StdRng::seed_from_u64(
            self.shuffle_seed.unwrap(),
        ))));

        self
    }

    fn validate_split_ratios(&self) -> Result<(), VKMLError> {
        if self.split_ratios.is_empty() {
            return Err(VKMLError::InvalidSplitRatios {
                message: "Split ratios cannot be empty".to_string(),
            });
        }

        let sum: f32 = self.split_ratios.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(VKMLError::InvalidSplitRatios {
                message: format!("Split ratios must sum to 1.0, got {}", sum),
            });
        }

        if self.split_ratios.iter().any(|&ratio| ratio <= 0.0) {
            return Err(VKMLError::InvalidSplitRatios {
                message: "All split ratios must be positive".to_string(),
            });
        }

        Ok(())
    }

    pub fn num_splits(&self) -> usize {
        self.split_ratios.len()
    }

    pub fn calculate_split_sizes(&self, total_size: usize) -> Vec<usize> {
        let mut split_sizes = Vec::with_capacity(self.split_ratios.len());
        let mut remaining = total_size;

        // calculate sizes for all splits except the last one
        for &ratio in &self.split_ratios[..self.split_ratios.len() - 1] {
            let split_size = (total_size as f32 * ratio) as usize;
            split_sizes.push(split_size);
            remaining = remaining.saturating_sub(split_size);
        }

        // last split gets all remaining items which handles rounding
        split_sizes.push(remaining);

        split_sizes
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            prefetch_count: 4,
            batch_size: 32,
            split_ratios: vec![0.8, 0.1, 0.1], // train, test, val
            sort_dataset: false,
            shuffle_seed: None,
            rng: None,
            drop_last: true,
        }
    }
}
