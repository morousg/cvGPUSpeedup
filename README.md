# cvGPUSpeedup

Every memory read, is an opportunity for compute.

With this idea in mind, this library wants to make OpenCV code run faster on the GPU. Especially for typical pre and post processing operations for DL networks.

The current code, implements operations that can preserve the Grid structure across different "kernels" or how we call them "operations". We do not discard to explore more complex grid patterns in the future, as hardware support for thread block communication improves.

This project is in it's infancy. It is a header-based C++/CUDA library, with a dual purpose:
1. To create a set of fusionable \_\_device\_\_ functions, that can be compiled with nvcc and the cuda runtime libraries, with no more dependencies. (namespace fk) 
2. To be able to use OpenCV-like code in the GPU, with OpenCV objects, with far more performance in some cases. (namespace cvGS)

The first main focus is on the transform pattern, with an incomplete set of basic arithmetic operations to be performed on cv::cuda::GpuMat objects.

In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc and at least C++17 support. We are testing it with CUDA version 11.8, on compute capabilities 7.5 and 8.6.

Let's see an example in OpenCV:

![alt text](https://github.com/morousg/cvGPUSpeedup/blob/98a268319b97955bf6d1fe0f3a611e0ea82f9d7d/OpenCVversion.png)

Now, same functionality but with cvGPUSpeedup and kernel execution being 38x times faster:

![alt text](https://github.com/morousg/cvGPUSpeedup/blob/2e9bfb1410c7dd8fb1bc4de279466637881b8843/cvGPUSpeedupVersion.png)

The cvGPUSpeedup version, will do the same, but with a single CUDA kernel, and execute up to 38x times faster, for 50 crops of an image.

# Benchmarks

The library has some unit tests that can be additionally used as benchmarks. When generating benchmark results, they show always positive speedups ranging from 2x to 1000x (in an RTX A2000). The Speedup is going to be greater the more kernels you are fusing, and the smaller those kernels are in terms of both compute operations and grid size. 

# cvGPUSpeedup at Mediapro

We used cvGPUSpeedup at AutomaticTV (Mediapro) for the preprocessing of Deep Neural Networks, and we obtained speedups of up to 167x compared to OpenCV-CUDA. At AutomaticTV we are developing DL networks and will continue to add functionality to cvGPUSPeedup.

<img src="https://github.com/morousg/cvGPUSpeedup/blob/main/images/NSightSystemsTimeline1.png" />

In the image above, we show two NSight Systems timelines, where before the execution of the neural network, we have to do some crops, resize, normalization and split of the color channels. 

In the case of OpenCV-CUDA, despite using the GPU you can see that OpenCV is launching many small kernels. This is wasting a lot of compute time in scheduling and memory accesses. You can even see some small Device to Device copies, which the DL programmers thought they needed.

With cvGPUSpeedup since the syntax is pretty similar to OpenCV, and all the parameters passed are OpenCV types, they managed to do the same operations but in 1/167th of the time, and reduced the amount of memory required in the GPU.

<img src="https://github.com/morousg/cvGPUSpeedup/blob/main/images/NsightSystemsTimeline2.png" />

In this other case, we are updating a temporal Tensor of 15 images, with a new image that needs to be resized and normalized, and other 14 images that where normalized in previous iterations, that need to be split to planar mode and copied in diferent positions of the temporal Tensor. Some CUDA threads will be doing the normalization, and some others will be just copying the old images, all in parallel.

As you can see, the resulting performance makes the pre-processing virtually free, when before it was more than 25% of the total time for the inference.

If you are interested in investing in cvGPUSpeedup development for your own usage, please contact us at oamoros@mediapro.tv
