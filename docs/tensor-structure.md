# Cranberry Tensor Structure Scheme

## Python Components

The Python components provide a logical representation of Tensors, including:
- Tensor
- Shape
- Strides
- Offset
- Device
- Dtype

These components don't handle the actual data; instead, they manage abstract Tensor operations.
Using Shape, Strides, and Offset, we define and slice operations, which are then passed to the kernels in the Rust components.

### TODO
- Implement an algorithm to access physical addresses using Shape, Strides, and Offset
    - Determine the physical index corresponding to a given logical index
    - If certain indices are contiguous (i.e., adjacent in the physical layer), identify the range $[l, r]$
    - The task of dividing operations into kernels is handled entirely here. It is not addressed in the Rust Components.
- Resources:
    - PyTorch internals: http://blog.ezyang.com/2019/05/pytorch-internals/
    - EurekaLabsAI/tensor: https://github.com/EurekaLabsAI/tensor

## Rust Components

The Rust components are responsible for managing the physical aspects of Tensors, specifically:
- Storage
- Kernel code for performing operations on Storage

### Why Rust?
- Easier memory management compared to C/C++
- High performance

### TODO
- Implement CPU, Metal, and CUDA operations for Storage
- Resources:
    - A Taste of GPU Compute (by Jane Street): https://youtu.be/eqkAaplKBc4?si=a03WY4QUKlLtOmck
    - gpu.cpp: https://github.com/AnswerDotAI/gpu.cpp
    - huggingface/candle: https://github.com/huggingface/candle
    - cuda-mode/lectures: https://github.com/cuda-mode/lectures