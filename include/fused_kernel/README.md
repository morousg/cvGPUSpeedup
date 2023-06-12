This folder contains the FusedKernel library, which can be used independently of OpenCV, and compiled with nvcc.

It is the building grounds of cvGPUSpeedup. In fact, cvGPUSpeedup is a wrapper arround FusedKernel, made to make it easy to use FusedKernel with OpenCV objects.

The goal is to allow programmers that are used to OpenCV, to very easily and intuituvely be able to use the FusedKernel library.
