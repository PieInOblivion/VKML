# VKML

This library contains high-level abstractions to make ML model development and usage easy and compute efficient.

## Project Priorities
1. Universal compute utilisation (Leverages any available hardware combination)
2. High heterogeneous compute efficiency
3. Predictable and consistent performance
4. Ease of use

## Overview
This project was inspired by research demonstrating a Fast Fourier Transform Vulkan implementation having "comparable or better performance" than CUDA (as demonstrated in [this IEEE paper](https://ieeexplore.ieee.org/document/10036080)).

As specific Vulkan ML extensions gradually evolve into standardised specifications and extensions, they will be implemented in this project. Currently it is working with solely shader computations, however this is changing rapidly (see [Vulkan Usage](#vulkan-usage)).

The project aims to provide abstractions at a level similar to PyTorch with default usage of:
- Multi-vendor GPU systems (Nvidia, AMD, Intel, Qualcomm, and more)
- GPU and CPU model training and inference

The proof of concept goal for this project will be met when we are able to benchmark a simple MLP model against it's PyTorch equivalent, while remaining an extensible framework for future development.

## Current Implementation Details (Assumptions, Descisions and Todo's)

### Overall Todo's
* Automatic workgroup decision making
* JIT like Descriptor Set Layout, Descriptor Sets, Pipeline Layout, Descriptor Pool optimisation implementation
  * The current implementation is still from early basic gpu testing
* Backwards Pass
* Multiple data formats
* Dataloader trait interface refactor

### Image Loading
* Current proof of concept implementation stores all file names in memory
  * Raw filesystems typically don't store file counts and aren't sorted, so we provide users that option for replicatability
  * Direct filesystem read into per batch means end files can never be in the first batch. Requires preread and store of filesystem
  * Future support planned for CSV and other formats
    * This will stop the need for prereading directory
  * Raw binary support planned

### Thread Pool Implementation
* Currently created once, leading to single-threaded usage creating a whole pool of one worker
  * This means tasks are loaded in advance, requiring more memory than running without a work queue
* Thread pool will be implemented as an option in future
* Current batch processing generates entire batch before submitting work
  * Could benefit from periodic queue flushing instead of sequential generate -> push -> work pattern

### GPU Management
* Currently assumes all GPU memory is free
  * Will implement VK_EXT_memory_budget in future (commonly implemented extension)
  * Final implementation will track own usage and initial usage from other processes
    * Will include configurable threshold (e.g., 95% of free memory)
* GPU filtering currently checks compute capability
  * Future investigation needed for non-compute flag GPUs
* GPU-to-GPU movement currently routes through CPU
  * Need to investigate Vulkan device pools
  * Research needed on VK shared memory pool extensions

### Architecture Decisions
* Model, Layer, Tensor etc. act as descriptors only
  * Allows the compute manager to handle all data and memory
  * Large seperation between blueprint layers and final tensor DAG
* ImageBatch to f32 function assumes little endian storage
* Current GPU memory calculations:
  * Don't account for allocation overhead (acceptable with safe memory threshold)
  * Don't track CPU memory requirements
* Model storage is sequential in memory
  * Prevents small layers being stored out of order on multi-device compute configurations
  * Avoids unnecessary CPU transfers
* Current compute implementation:
  * Sends and waits for single GPU commands
  * Future improvement: multiple simultaneous commands using threadpool or native Vulkan solution

### Vulkan Usage
* Vendor specific extensions become standard extensions depending on adoption. As of 2025, ARM appears to be focusing on adding ML specific extension to Vulkan
  * As of 1.3.300 [VK_NV_cooperative_matrix](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_matrix.html)
  * As of 1.4.317 [VK_EXT_shader_float8](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_shader_float8.html)
  * As of 1.4.319 [VK_ARM_data_graph](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_ARM_data_graph.html)

## Building
* Requires [glslc](https://github.com/google/shaderc) in PATH to compile shaders

## References

### Vulkan Resources
* [Cooperative Matrix Performance](https://github.com/jeffbolznv/vk_cooperative_matrix_perf)
* [Vulkan Tutorial PDF](https://vulkan-tutorial.com/resources/vulkan_tutorial_en.pdf)
* [Rust Vulkan Tutorial](https://github.com/unknownue/vulkan-tutorial-rust)
* [Ash-rs](https://github.com/ash-rs/ash)
* [Vulkano](https://github.com/KyleMayes/vulkanalia)
* [Vulkanalia](https://github.com/KyleMayes/vulkanalia)
* [VkFFT](https://github.com/DTolm/VkFFT)
* [IEEE Paper](https://ieeexplore.ieee.org/document/10036080)

### Related Projects
* [Burn](https://github.com/tracel-ai/burn)
* [Candle](https://github.com/huggingface/candle)
* [AdaptiveCpp](https://adaptivecpp.github.io/AdaptiveCpp/)
