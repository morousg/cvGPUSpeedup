# FusedKernel library

This folder contains the FusedKernel library, which can be used independently of OpenCV, and compiled with nvcc.

It is the building grounds of cvGPUSpeedup. In fact, cvGPUSpeedup is a wrapper arround FusedKernel, made to make it easy to use FusedKernel with OpenCV objects.

The goal is to allow programmers that are used to OpenCV, to very easily and intuituvely be able to use the FusedKernel library.

## Fusion and inclusion

The way the FusedKernel library is implemented, allows not only to use the already implemented Operations and data types like Ptr2D or Tensor, but also the fusion can be performend using any code that conforms to the FusedKernel interface (the DeviceFunction structs and the operate function types) and those operations can use any data type that the user wants to use.

This was done in purpose to make it easier to join efforts with other libraries that already exist and are also OpenSource and want to take advantage of the FusedKernel strategy.
### Horizontal Fusion

This fusion technique is widely known and used. It is based on the idea of processing several data planes in parallel, with the same CUDA kernel. For that, we use the blockIdx.z, to distinguish between thread planes and data planes.

This is usually very beneficial when each plane is very small, and the resulting 2D Grid is not taking advantage of the GPU memory bandwidth.

In the case of DivergentBatchGridPattern, we also allow to use different code seqüences that may use a different amount of threads, on each plane. That last case, can allow for hawing different warps using different SM hardware components in parallel, contributiong to the overall speedup.

### Vertical Fusion

This fusion, is usually limited to having a kernel, that is configurable up to a certain level. In our case, we are abstrating away the thread behavior from the actual functionality, and allowing to fuse almost every kernel possible, without having to rewrite neither the thread handling, nor the functionality. You only have to combine code, in the different ways that the code can be combined.

For Memory Bound kernels, this fusion is bringing most of the performance improvements, since adding more functions to the kernel, will not increase the execution time, up to a limit where the kernel becomes compute bound.

Not only that, but the way the code is written, the nvcc compiler will treat the consecutive functions as if you where writting the code in one line, adding all sorts of optimizations. This can be seen by compiling the code in Release mode, or in Debug mode. The performance difference is abismal.

### Compute only what you need

This is an optimization that can already be used with the current code, but will further increase the use cases, when addind more Operations.

The idea, is aplicable for situations where you have a big plane, from which you will only use a subset of the data. If you need to transform that plane into something different, before doing the operation that will read the subset of elements, you can us vertical fusion in order to have a single kernel, that will read only what it needs, and apply to it all the transformations needed.

For example, let's assume that you receive an image in YUV420_NV12 format, and you need to crop a region of this image, then convert the pixels to RGB, then resize the crop, normalize it to floating point values from 0 to 1, and store the resulting image in RGB planar format. Usually, this would lead to many kernels, one after the other. The first kernel that converts to RGB, will convert the full image, and write the result to memory. Instead, with the Fused Kernel library (when the YUV conversion functionality is integrated), it will be possible to create a Fused Kernel in a few lines, that will only read the YUV data for the pixels required by the interpolation process, in the resize. And all the steps will be performed using GPU registers, until the last step where we will finally write into GPU ram memory.

Obviously, this is way faster than the conventional way of programming.

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, they can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of a DeviceFunction struct.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
