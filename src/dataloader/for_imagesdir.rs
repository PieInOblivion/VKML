use image;
use image::ColorType;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::thread_pool::thread_pool::ThreadPool;
use crate::thread_pool::worker::{WorkResult, WorkType};

use super::config::DataLoaderConfig;
use super::data_batch::DataBatch;
use super::dataloader::{DataLoader, DatasetSplit, SourceFormat};
use super::error::VKMLEngineError;

impl From<ColorType> for SourceFormat {
    fn from(color_type: ColorType) -> Self {
        match color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => SourceFormat::U8,
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => {
                SourceFormat::U16
            }
            ColorType::Rgb32F | ColorType::Rgba32F => SourceFormat::F32,
            _ => panic!("Unsupported color type"),
        }
    }
}

pub struct DirectoryImageLoader {
    dir: PathBuf,
    dataset: Vec<Box<str>>,
    dataset_indices: Vec<usize>,
    valid_extensions: HashSet<String>,
    image_width: u32,
    image_height: u32,
    image_channels: u32,
    image_bytes_per_pixel: u32,
    image_bytes_per_image: usize,
    image_total_bytes_per_batch: usize,
    image_color_type: ColorType,
    image_source_format: SourceFormat,
    config: DataLoaderConfig,
    thread_pool: Arc<ThreadPool>,
}

impl DirectoryImageLoader {
    pub fn new(
        dir: &str,
        config: Option<DataLoaderConfig>,
        thread_pool: Arc<ThreadPool>,
    ) -> Result<Self, VKMLEngineError> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(VKMLEngineError::DirectoryNotFound(dir.to_string()));
        }

        let valid_extensions = image::ImageFormat::all()
            .flat_map(|format| format.extensions_str())
            .map(|ext| ext.to_string())
            .collect();

        let mut loader = DirectoryImageLoader {
            dir: path.to_owned(),
            dataset: Vec::new(),
            dataset_indices: Vec::new(),
            valid_extensions,
            image_width: 0,
            image_height: 0,
            image_channels: 0,
            image_bytes_per_pixel: 0,
            image_bytes_per_image: 0,
            image_total_bytes_per_batch: 0,
            image_color_type: ColorType::Rgb32F,
            image_source_format: SourceFormat::F32,
            config: config.unwrap_or_default(),
            thread_pool,
        };

        loader.load_dataset()?;
        loader.scan_first_image()?;

        Ok(loader)
    }

    fn load_dataset(&mut self) -> Result<(), VKMLEngineError> {
        // TODO: Use thread pool and batch work
        self.dataset = std::fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .filter(|entry| self.is_valid_extension(&entry.path()))
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.to_owned().into_boxed_str())
            })
            .collect();

        if self.dataset.is_empty() {
            return Err(VKMLEngineError::EmptyDataset);
        }

        // read_dir does not guarantee consistancy or sorting of any kind since filesystems don't either
        // Useful if trying the same dataset on different systems or filesystems
        if self.config.sort_dataset {
            self.dataset.sort_unstable();
        }

        // .collect() can use size_hint from std::ops::Range
        self.dataset_indices = (0..self.dataset.len()).collect();

        self.config.rng = if self.config.shuffle {
            if self.config.shuffle_seed.is_none() {
                self.config.shuffle_seed = Some(rand::rng().random());
            }
            Some(Arc::new(Mutex::new(StdRng::seed_from_u64(
                self.config.shuffle_seed.unwrap(),
            ))))
        } else {
            None
        };

        self.shuffle_whole_dataset()?;

        Ok(())
    }

    fn scan_first_image(&mut self) -> Result<(), VKMLEngineError> {
        let first_image_path = PathBuf::from(&self.dir).join(&*self.dataset[0]);
        let img = image::open(first_image_path)?;
        self.image_width = img.width();
        self.image_height = img.height();
        self.image_channels = img.color().channel_count() as u32;
        self.image_bytes_per_pixel = img.color().bytes_per_pixel() as u32;

        self.image_color_type = img.color();
        self.image_source_format = SourceFormat::from(img.color());

        self.image_bytes_per_image = self.image_width as usize
            * self.image_height as usize
            * self.image_bytes_per_pixel as usize;
        self.image_total_bytes_per_batch = self.image_bytes_per_image * self.config.batch_size;
        Ok(())
    }

    fn is_valid_extension(&self, path: &PathBuf) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.valid_extensions.contains(&ext.to_lowercase()))
            .unwrap_or(false)
    }

    fn get_split_sizes(&self) -> (usize, usize, usize) {
        let total_size = self.dataset.len();
        let train_size = (total_size as f32 * self.config.train_ratio) as usize;
        let test_size = (total_size as f32 * self.config.test_ratio) as usize;
        let val_size = total_size - train_size - test_size;
        (train_size, test_size, val_size)
    }
}

impl DataLoader for DirectoryImageLoader {
    type BatchDataReference = Vec<PathBuf>;

    fn get_batch_reference(
        &self,
        split: DatasetSplit,
        batch_number: usize,
    ) -> Option<Self::BatchDataReference> {
        let (train_size, test_size, _) = self.get_split_sizes();
        let (start_index, end_index) = match split {
            DatasetSplit::Train => (0, train_size),
            DatasetSplit::Test => (train_size, train_size + test_size),
            DatasetSplit::Validation => (train_size + test_size, self.dataset.len()),
        };

        let split_size = end_index - start_index;
        let batch_start = batch_number * self.config.batch_size;

        if batch_start >= split_size {
            return None;
        }

        let batch_end = (batch_start + self.config.batch_size).min(split_size);
        let is_last_batch = batch_end == split_size;

        if self.config.drop_last
            && is_last_batch
            && (batch_end - batch_start) < self.config.batch_size
        {
            return None;
        }

        let indices = &self.dataset_indices[start_index + batch_start..start_index + batch_end];
        let paths = indices
            .iter()
            .map(|&idx| self.dir.join(&*self.dataset[idx]))
            .collect();

        Some(paths)
    }

    fn shuffle_whole_dataset(&mut self) -> Result<(), VKMLEngineError> {
        let mut rng = self
            .config
            .rng
            .as_ref()
            .ok_or(VKMLEngineError::RngNotSet)?
            .lock()
            .map_err(|_| VKMLEngineError::RngLockError)?;
        self.dataset_indices.shuffle(&mut *rng);
        Ok(())
    }

    fn shuffle_individual_datasets(&mut self) -> Result<(), VKMLEngineError> {
        let (train_size, test_size, _) = self.get_split_sizes();

        let mut rng = self
            .config
            .rng
            .as_ref()
            .ok_or(VKMLEngineError::RngNotSet)?
            .lock()
            .map_err(|_| VKMLEngineError::RngLockError)?;

        self.dataset_indices[0..train_size].shuffle(&mut *rng);
        self.dataset_indices[train_size..train_size + test_size].shuffle(&mut *rng);
        self.dataset_indices[train_size + test_size..].shuffle(&mut *rng);
        Ok(())
    }

    fn create_batch_work(&self, batch_number: usize, paths: Vec<PathBuf>) -> WorkType {
        let batch_size = paths.len();
        WorkType::LoadImageBatch {
            batch_number,
            paths,
            image_total_bytes_per_batch: self.image_total_bytes_per_batch,
            image_bytes_per_image: self.image_bytes_per_image,
            image_color_type: self.image_color_type,
            batch_size,
            thread_pool: self.get_thread_pool(),
        }
    }

    fn process_work_result(&self, result: WorkResult, expected_batch: usize) -> DataBatch {
        match result {
            WorkResult::LoadImageBatch {
                batch,
                batch_number,
            } => {
                debug_assert_eq!(batch_number, expected_batch);
                batch
            }
            _ => panic!("Unexpected work result type"),
        }
    }

    fn get_config(&self) -> &DataLoaderConfig {
        &self.config
    }

    fn get_thread_pool(&self) -> Arc<ThreadPool> {
        self.thread_pool.clone()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
