use image::{self, ColorType};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use vkml::dataloader::{
    config::DataLoaderConfig,
    dataloader::DataLoader,
    error::VKMLError,
    info::print_dataset_info,
    par_iter::DataLoaderParIter,
};
use onnx_extractor::DataType;
use zero_pool::ZeroPool;

impl From<ColorType> for DataType {
    fn from(color_type: ColorType) -> Self {
        match color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => {
                DataType::Uint8
            }
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => {
                DataType::Uint16
            }
            ColorType::Rgb32F | ColorType::Rgba32F => DataType::Float,
            _ => panic!("Unsupported color type"),
        }
    }
}

pub struct ImagesDirDataLoader {
    dir: PathBuf,
    dataset: Vec<Box<str>>,
    dataset_indices: Vec<usize>,
    valid_extensions: HashSet<String>,
    image_width: u32,
    image_height: u32,
    image_channels: u32,
    image_bytes_per_pixel: u32,
    image_bytes_per_image: usize,
    image_color_type: ColorType,
    image_data_type: DataType,
    config: DataLoaderConfig,
    thread_pool: ZeroPool,
}

impl ImagesDirDataLoader {
    pub fn new(
        dir: &str,
        config: Option<DataLoaderConfig>,
        thread_pool: ZeroPool,
    ) -> Result<Self, VKMLError> {
        let path = Path::new(dir);
        if !path.exists() {
    return Err(VKMLError::Generic(format!(
        "Directory not found: {}",
        dir
    )));
}

        let valid_extensions = image::ImageFormat::all()
            .flat_map(|format| format.extensions_str())
            .map(|ext| ext.to_string())
            .collect();

        let config = config.unwrap_or_default().build();

        let mut loader = ImagesDirDataLoader {
            dir: path.to_owned(),
            dataset: Vec::new(),
            dataset_indices: Vec::new(),
            valid_extensions,
            image_width: 0,
            image_height: 0,
            image_channels: 0,
            image_bytes_per_pixel: 0,
            image_bytes_per_image: 0,
            image_color_type: ColorType::Rgb32F,
            image_data_type: DataType::Float,
            config,
            thread_pool,
        };

        loader.load_dataset()?;
        loader.scan_first_image()?;

        Ok(loader)
    }

    fn load_dataset(&mut self) -> Result<(), VKMLError> {
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
    return Err(VKMLError::Generic(
        "No images found in the dataset".to_string()
    ));
}

        if self.config.sort_dataset {
            self.dataset.sort_unstable();
        }

        self.dataset_indices = (0..self.dataset.len()).collect();
        self.shuffle_whole_dataset();

        Ok(())
    }

    fn scan_first_image(&mut self) -> Result<(), VKMLError> {
        let first_image_path = PathBuf::from(&self.dir).join(&*self.dataset[0]);
        let img = image::open(first_image_path)?;
        self.image_width = img.width();
        self.image_height = img.height();
        self.image_channels = img.color().channel_count() as u32;
        self.image_bytes_per_pixel = img.color().bytes_per_pixel() as u32;
        self.image_color_type = img.color();
        self.image_data_type = DataType::from(img.color());
        self.image_bytes_per_image = self.image_width as usize
            * self.image_height as usize
            * self.image_bytes_per_pixel as usize;
        Ok(())
    }

    fn is_valid_extension(&self, path: &PathBuf) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.valid_extensions.contains(&ext.to_lowercase()))
            .unwrap_or(false)
    }
}

impl DataLoader for ImagesDirDataLoader {
    fn len(&self) -> (usize, Vec<usize>) {
        let total = self.dataset.len();
        let split_sizes = self.config.calculate_split_sizes(total);
        (total, split_sizes)
    }

    fn get_batch_indices(&self, split_idx: usize, batch_idx: usize) -> Option<Vec<usize>> {
        let (_, split_sizes) = self.len();
        if split_idx >= split_sizes.len() {
            return None;
        }

        let split_start: usize = split_sizes[..split_idx].iter().sum();
        let split_size = split_sizes[split_idx];
        let batch_start = batch_idx * self.config.batch_size;

        if batch_start >= split_size {
            return None;
        }

        let batch_end = (batch_start + self.config.batch_size).min(split_size);
        let is_last_batch = batch_end == split_size;

        if self.config.drop_last && is_last_batch && (batch_end - batch_start) < self.config.batch_size {
            return None;
        }

        let indices: Vec<usize> = (split_start + batch_start..split_start + batch_end)
            .map(|i| self.dataset_indices[i])
            .collect();

        Some(indices)
    }

    fn bytes_per_item(&self) -> usize {
        self.image_bytes_per_image
    }

    fn batch_data_type(&self) -> DataType {
        self.image_data_type
    }

    fn get_batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn get_item_bytes(&self, index: usize) -> Vec<u8> {
        let path = self.dir.join(&*self.dataset[index]);
        let img = image::open(path).expect("Failed to load image");
        img.as_bytes().to_vec()
    }

    fn shuffle_whole_dataset(&mut self) {
    let mut rng = self.config.rng.as_ref()
        .expect("RNG should always be initialized")
        .lock()
        .expect("Failed to acquire RNG lock");
    self.dataset_indices.shuffle(&mut *rng);
}

    fn shuffle_individual_dataset(&mut self, split_idx: usize) {
    let (_, split_sizes) = self.len();
    assert!(
        split_idx < split_sizes.len(),
        "Invalid split index: {}, only {} splits available",
        split_idx, split_sizes.len()
    );

    let split_start: usize = split_sizes[..split_idx].iter().sum();
    let split_end = split_start + split_sizes[split_idx];

    let mut rng = self.config.rng.as_ref()
        .expect("RNG should always be initialized")
        .lock()
        .expect("Failed to acquire RNG lock");

    self.dataset_indices[split_start..split_end].shuffle(&mut *rng);
}

    fn get_config(&self) -> &DataLoaderConfig {
        &self.config
    }

    fn get_thread_pool(&self) -> &ZeroPool {
        &self.thread_pool
    }
}

fn main() {
    // Example usage
    let thread_pool = ZeroPool::new();

    let config = DataLoaderConfig {
        batch_size: 32,
        split_ratios: vec![0.7, 0.2, 0.1], // train, test, val
        prefetch_count: 2,
        ..Default::default()
    };

    let loader = ImagesDirDataLoader::new(
        "./data/images",
        Some(config),
        thread_pool,
    ).expect("Failed to create dataloader");

    print_dataset_info(&loader);

    // Iterate through training batches
    println!("Processing training batches...");
    for (i, batch) in loader.par_iter(0).enumerate() {
        println!(
            "Batch {}: {} bytes, dtype: {:?}", 
            i, 
            batch.len(),
            batch.data_type()
        );
        
        // Example: Convert to f32 if needed for model
    if batch.data_type() != DataType::Float {
            let f32_data = batch.to_f32();
            println!("  Converted {} elements to f32", f32_data.len());
        }
    }
}