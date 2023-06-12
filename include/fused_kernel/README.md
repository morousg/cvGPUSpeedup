# FusedKernel library

This folder contains the FusedKernel library, which can be used independently of OpenCV, and compiled with nvcc.

It is the building grounds of cvGPUSpeedup. In fact, cvGPUSpeedup is a wrapper arround FusedKernel, made to make it easy to use FusedKernel with OpenCV objects.

The goal is to allow programmers that are used to OpenCV, to very easily and intuituvely be able to use the FusedKernel library.

## Fusion and inclusion

The way the FusedKernel library is implemented, allows not only to use the already implemented Operations and data types like Ptr2D or Tensor, but alse the fusion can be performend using any code that conforms to the FusedKernel interface (the DeviceFunction structs and the operate function types) and those operations can use any data type that the user wants to use.

This was done in purpose to make it easier to join efforts with other libraries that already exist and are also OpenSource and want to take advantage of the FusedKernel strategy.

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, they can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of a DeviceFunction struct.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
