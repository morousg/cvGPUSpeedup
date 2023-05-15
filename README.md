# cvGPUSpeedup

Every memory read, is an opportunity for compute.

Whith this idea in mind, this library wants to make OpenCV code run faster on the GPU. Especially for typical pre and post processing operations for DL networks.

The current code, implements operations that can preserve the Grid structure across different "kernels" or how we call them "operations". We do not discard to explore more complex grid patterns in the future, as hardware support for thread block communication improves.

This project is in it's infancy. It is a header-based C++/CUDA library, with a dual purpose:
1. To create a set of fusionable __device__ functions, that can be compiled only with nvcc and the cuda runtime libraries. (namespace fk) 
2. To be able to use OpenCV-like code in the GPU, with far more performance in some cases. (namespace cvGS)

The first main focus is on the transform operation, with an incomplete set of basic arithmetic operations to be performed on cv::cuda::GpuMat objects.

In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc and at least C++17 support. We are testing it with CUDA version 11.8, on compute capabilities 7.5 and 8.6.

Let's see an example in OpenCV:

![alt text](https://github.com/morousg/cvGPUSpeedup/blob/98a268319b97955bf6d1fe0f3a611e0ea82f9d7d/OpenCVversion.png)

Now, same functionality but with cvGPUSpeedup and kernel execution being 38x times faster:

![alt text](https://github.com/morousg/cvGPUSpeedup/blob/98a268319b97955bf6d1fe0f3a611e0ea82f9d7d/cvGPUSpeedupVersion.png)

The cvGPUSpeedup version, will do the same, but with a single CUDA kernel, and execute up to 38x time faster, for 50 crops of an image.

A performance tip. Do not use static arrays inside your struct. The nvcc compiler will do bad things that will heavily affect the resulting kernel performance.
