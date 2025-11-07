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
* Graph optimisaion pass; fusion etc
* VK_NV_cooperative_matrix2
* Interface for manual instruction and tensor modification into model and/or tensor graphs
* Backwards Pass
* Dynamic graph support (Conditionals, Loops, etc)
* Implementing more instructions/operations for more datatypes
* Multi-threading for CPU operation/instructions; matmul, gemm, etc. Currently they serve only as references and proof of concept implementations

### Thread Pool Implementation
* One global thread pool is used throughout the entire process, zero-pool
* This means that in most cases, bar some older dgpu specifications that require staged allocation logic, the entire process is multi-threaded where possible and lock free.

### GPU Management
* Memory tracking implemented using VK_EXT_memory_budget when available
  * Tracks both self usage and initial usage from other processes
  * Configurable threshold (default 95% of available memory)
  * Multi-threaded allocation
* Automatic model placement across available devices (GPUs and CPU)
  * Automatically creates transfer operations when model is split across devices
  * Handles host-visible vs device-local memory requirements
* GPU features are taken into account, and performance features are toggled as supported on a per device level
  * No vendor specific extensions used (for now)
* GPU-to-GPU movement currently routes through CPU
  * Need to investigate Vulkan device pools
  * Research needed on VK shared memory pool extensions

### Architecture Decisions
* Model, Layer, Tensor etc. act as descriptors/blueprints only
  * Allows the compute manager to handle all data and memory
  * Large separation between blueprint layers and final tensor DAG
* Memory tracking:
  * GPU memory tracked atomically with VK_EXT_memory_budget integration
  * CPU memory tracked with atomic counters
  * Pre-allocation checks ensure model fits before attempting allocation
* Zero-copy optimisations:
  * Model loading is full zero-copy
  * CPU allocations use zero-copy transfer when possible
  * GPU allocations use reference-based zero-copy transfer when possible
* Model storage is sequential in memory
  * Avoids unnecessary CPU transfers
* Current compute implementation:
  * All parallelisable work is done so, relying on gpu hardware scheduler
  * Parallel tensor allocation using thread pool

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
* [VkFFT IEEE Paper](https://ieeexplore.ieee.org/document/10036080)
* [Nvidia Recommended Do's and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts)

### Related Projects
* [Burn](https://github.com/tracel-ai/burn)
* [Candle](https://github.com/huggingface/candle)
* [tinygrad](https://github.com/tinygrad/tinygrad)
* [AdaptiveCpp](https://adaptivecpp.github.io/AdaptiveCpp/)
