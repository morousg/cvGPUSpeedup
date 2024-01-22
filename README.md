# cvGPUSpeedup

Every memory read, is an opportunity for compute.

With this idea in mind, this library wants to make OpenCV code run faster on the GPU. Especially for typical pre and post processing operations for DL networks.

The current code implements some OpenCV-CUDA functions and other functions that are not available in OpenCV, in a way that they can be fused in a single CUDA Kernel, making the resulting performance way better. For the final user, the way of using those functions it's very similar to the OpenCV ones. The main difference is that the functions do not execute the code in the GPU, but return an struct that will conaint the parameters and the code. When you have all the operations, the only remaining thing to do, is to call cvGS::executeOperations passing a cuda stream, and the operations as parameters in the order you want them to be executed. The result of the first operation will be the input of the next operation and so on, until the nth operation whose input will be the result of the nth - 1 operation.

In addition to the performance gains, this reduces the number of cv::cuda::GpuMat objects required, since you will only need an input GpuMat object for the first operation, and an output GpuMat object for the last operation. This is just the basic functionality, there are more complex options. Unfortunatly, there is no documentation at this time, so the best way to check what is possible is to look at the source code in the tests folder.

This project is early stages and continuosly evolving to provide further performance enhancements. It is a header-based C++/CUDA library, with several goals:
1. To provide a set of fusionable \_\_device\_\_ functions, built only with nvcc and the cuda runtime libraries, (namespace fk) 
2. Enabling OpenCV-like code in the GPU, with OpenCV objects, with far more performance in some cases. (namespace cvGS)

The first main focus is on the transform pattern, with an incomplete set of basic arithmetic operations to be performed on cv::cuda::GpuMat objects.

  
## Tested hw/sw
*  Cuda SDK 11.8 and 12.1
*  OS Windows 11 22H2 with drivers from 516.94 to 546.17.
*  Ubuntu 22.04 (both native and under WSL2 enviroment)   
*  Compute capabilities 7.5 (Turing), 8.6 (Ampere), 8.9 (ADA Lovelace)
*  All systems with x86_64 cpu architecture

## Using the library
In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc (provided by the CUDA toolkit) and at least C++17 support (this is already set by the cmake project).

You can use the cmake install target to copy the headers to any desired path. You can also use the cmake exported target.

If you want to use the cvGS interface, along with OpenCV, both on windows and linux you will need to build opencv with cuda support. If you want to only use the fk (Fused Kernel) namespace, then you won't need OpenCV at all.

### OpenCV example
Let's see an example in OpenCV:
```cpp
// OpenCV version
void testOpenCV()
{
    constexpr int MAX_DETECTIONS = 50;
    std::array<cv::cuda::GpuMat, MAX_DETECTIONS> crops;
    // Fill the crops array with 50 crops of any size, from a source image.

    cv::Scalar subtract_val(1, 4, 6);
    cv::Scalar divide_val(2, 8, 1);

    cv::cudaStream stream;
    cv::Size resDims(64, 128);
    std::array<std::vector<cv::cuda::GpuMat>, MAX_DETECTIONS> cv_output;
    for (int detection = 0; detection < MAX_DETECTIONS; detection++) {
        for (int i = 0; i < CV_MAT_CN(CV_32FC3); i++) {
            cv_output[detection].emplace_back(resDims, CV_32FC1);
        }
    }

    // All this memory required by OpenCV only
    cv::cuda::GpuMat resized(resDims, CV_8UC3);
    cv::cuda::GpuMat float3Image(resDims, CV_32FC3);
    cv::cuda::GpuMat float3Image2(resDims, CV_32FC3);

    double alpha = 0.5;

   // 50 times 5 kernels, 250 kernels in total
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        cv::cuda::resize(crops[i], resized, resDims, 0., 0., cv::INTER_LINEAR, stream);
        resized.convertTo(float3Image, CV_32FC3, alpha, stream);
        cv::cuda::subtract(float3lmage, subtract_val, float3lmage2, cv::noArray(), -1, stream);
        cv::cuda::divide(float3lmage2, divide_val, float3lmage, 1.0, -1, stream);
        cv::cuda::split(float3Image, cv_output[i], stream);
   }
    stream.waitForCompletion();
}
``` 
### cvGPUSpeedup example
Now, same functionality but with cvGPUSpeedup and kernel execution being 38x times faster:

```c++
// cvGPUSpeedup version
void testcvGPUSpeedup()
{
    constexpr int MAX_DETECTIONS = 50;
    std::array<cv::cuda::GpuMat, MAX_DETECTIONS> crops;
    // Fill the crops array with 50 crops of any size, from a source image.

    cv::Scalar subtract_val(1, 4, 6);
    cv::Scalar divide_val(255, 255, 255);
    
    cv::cudaStream stream;
    cv::Size resDims(64, 128);
    cv::cuda::GpuMat output(MAX_DETECTIONS, resDims.width * resDims.height * CV_MAT_CN(CV_32FC3), CV_32FC1);
    double alpha = 0.5;

    // Asume we got the maximum number of detections
    int activeDetections = 50;
    // single kernel, 38x faster than OpenCV in RTX A2000 12GB
    cvGS::executeOperations(stream,
                            cvGS::resize<CV_8UC3, cv::INTER_LINEAR, MAX_DETECTIONS>(crops,resDims,activeDetections)),
                            cvGS::convertTo<CV_8UC3, CV_32FC3>(),
                            cvGS::multiply<CV_32FC3>(cv::Scalar(alpha,alpha,alpha)),
                            cvGS::substract<CV_32FC3>(substract_val),
                            cvGS::divide<CV_32FC3>(divide_val),
                            cvGS::split<CV_32FC3>(output,resDims)
                            );

    stream.waitForCompletion();
}
```

The cvGPUSpeedup version, will do the same, but with a single CUDA kernel, and execute up to 38x times faster, for 50 crops of an image.

# Benchmarks

The library has some unit tests that can be additionally used as benchmarks. When generating benchmark results, they show always positive speedups ranging from 2x to 10000x (in an RTX A2000). The Speedup is going to be greater the more kernels you are fusing, and the smaller those kernels are in terms of both compute operations and grid size. 
## Variable size crop, resize and normalize

![OpenCV timeline](https://github.com/morousg/cvGPUSpeedup/blob/main/images/NSightSystemsTimeline1.png) 
   
In the image above, we show two NSight Systems timelines, where before the execution of the neural network, we have to do some crops, resize, normalization and split of the color channels. 
In the case of OpenCV-CUDA, despite using the GPU you can see that OpenCV is launching many small kernels. This is wasting a lot of compute time in scheduling and memory accesses. You can even see some small Device to Device copies, which the DL programmers thought they needed.
With cvGPUSpeedup since the syntax is pretty similar to OpenCV, and all the parameters passed are OpenCV types, they managed to do the same operations but in 1/167th of the time, and reduced the amount of memory required in the GPU.

## Temporal tensor (15 images)

![cvGPUSpeedup timeline](https://github.com/morousg/cvGPUSpeedup/blob/main/images/NsightSystemsTimeline2.png) 

In this other case, we are updating a temporal Tensor of 15 images, with a new image that needs to be resized and normalized, and other 14 images that where normalized in previous iterations, that need to be split to planar mode and copied in diferent positions of the temporal Tensor. Some CUDA threads will be doing the normalization, and some others will be just copying the old images, all in parallel.

As you can see, the resulting performance makes the pre-processing virtually free, when before it was more than 25% of the total time for the inference.

# Final words and contact
[Grup Mediapro](https://www.mediapro.tv) uses cvGPUSpeedup in the [AutomaticTV](https://www.automatic.tv) multicam live sports production system.  This product depends on custom Deep Neural Networks. Compared to vanilla OpenCV-CUDA implementation,  we obtained speedups of up to 167x in some cases.

If you are interested in investing in cvGPUSpeedup development for your own usage, please contact <oamoros@mediapro.tv>
