use super::dataloader::DataLoader;
use crate::dataloader::config::DataLoaderConfig;

pub fn print_dataset_info(dl: &impl DataLoader) {
    let (total_size, split_sizes) = dl.len();
    let config: &DataLoaderConfig = dl.get_config();

    println!("Dataset Information:");
    println!("-------------------");
    println!("Total size: {}", total_size);
    println!("Batch size: {}", config.batch_size);
    println!();

    for (i, (&split_size, &split_ratio)) in split_sizes.iter().zip(&config.split_ratios).enumerate()
    {
        let split_batches = dl.batches_in_split(i);
        let last_batch_size = split_size % config.batch_size;

        println!("Split {}:", i);
        println!("  Size: {} ({:.2}%)", split_size, split_ratio * 100.0);
        println!("  Batches: {}", split_batches);
        println!("  Last batch size: {}", last_batch_size);
        println!();
    }

    println!("Seed: {:?}", config.shuffle_seed);
}
