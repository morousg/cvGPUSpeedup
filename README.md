# cvGPUSpeedup

Every memory read, is an opportunity for compute.

Whith this idea in mind, this library wants to make OpenCV code run faster on the GPU. Especially the cudaarithm operations, in combination with others like type conversions, crops, resizes etc...

The current code, implements operations that can preserve the Grid structure across different "kernels" or how we call them "operations". We do not discard to explore more complex grid patterns in the future, as hardware support for thread block communication improves.

This project is in it's infancy. It is a header-based C++/CUDA library, with a dual purpose:
1. To create a set of fusionable __device__ functions, that can be compiled only with nvcc and the cuda runtime libraries. (namespace fk) 
2. To be able to use OpenCV-like code in the GPU, with far more performance in some cases. (namespace cvGS)

The first main focus is on the transform operation, with an incomplete set of basic arithmetic operations to be performed on cv::cuda::GpuMat objects.

In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc and at least C++14 support. We are testing it with CUDA version 11.8, on compute capabilities 7.5 and 8.6.

This is how the usage of cvGPUSpeedup looks like, compared to OpenCV:

![alt text](https://github.com/morousg/cvGPUSpeedup/blob/0acabe5354fb99fcc3d27ff5982b32d6c320bf15/cvGPUSpeedupExample.png)

The cvGPUSpeedup version, will do the same, but with a single CUDA kernel, and execute from 4x up to 7x faster, depending on the GPU, driver version, and OS.

Now imagine that you have to execute that same code, for 30 or 40 crops from a source image. Then the speedup can grow to 10x.

If you look at the code, it's not that difficult to add your own kernels into the game. You just need to create your struct, and make an operate_noret() version that will handle your struct parameters, and call your kernel as a __device__ function, with those parameters.

In order to "fuse" it with other kernels:
1. Instead of reading from device memory, use the value generated in the previous __device__ function, passed to your operate_noret function by the previous one.
2. Instead of writting to device memory, make the __device__ function to return the result.
3. Make your operate_noret function version, to pass the result as the input for the next operate_noret call.
4. To read the data, use one of the read memory patterns available as the first struct, or create your own.
5. To write the data, use one of the write memory patterns available as the last struct, or create your own.
