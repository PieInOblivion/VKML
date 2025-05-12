use crate::dataloader::config::DataLoaderConfig;

use super::dataloader::DataLoader;

pub fn print_dataset_info(dl: &impl DataLoader) {
    let total_size = dl.len();
    let config: &DataLoaderConfig = dl.get_config();

    let train_size = (total_size as f32 * config.train_ratio) as usize;
    let test_size = (total_size as f32 * config.test_ratio) as usize;
    let val_size = total_size - train_size - test_size;

    let train_batches = (train_size + config.batch_size - 1) / config.batch_size;
    let test_batches = (test_size + config.batch_size - 1) / config.batch_size;
    let val_batches = (val_size + config.batch_size - 1) / config.batch_size;

    println!("Dataset Information:");
    println!("-------------------");
    println!("Total size: {}", total_size);
    println!("Batch size: {}", config.batch_size);
    println!();
    println!("Train split:");
    println!(
        "  Size: {} ({:.2}%)",
        train_size,
        config.train_ratio * 100.0
    );
    println!("  Batches: {}", train_batches);
    println!("  Last batch size: {}", train_size % config.batch_size);
    println!();
    println!("Test split:");
    println!("  Size: {} ({:.2}%)", test_size, config.test_ratio * 100.0);
    println!("  Batches: {}", test_batches);
    println!("  Last batch size: {}", test_size % config.batch_size);
    println!();
    println!("Validation split:");
    println!(
        "  Size: {} ({:.2}%)",
        val_size,
        (1.0 - config.train_ratio - config.test_ratio) * 100.0
    );
    println!("  Batches: {}", val_batches);
    println!("  Last batch size: {}", val_size % config.batch_size);
    println!();
    println!("Shuffle: {}", config.shuffle);
    println!("Seed: {:?}", config.shuffle_seed);
}
