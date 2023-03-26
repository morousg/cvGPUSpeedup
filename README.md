# cvGPUSpeedup
Make OpenCV code run faster on the GPU

This is project is in it's infancy. It is a header-based C++/CUDA library, with the single purpose to be able to use OpenCV-like code in the GPU, with far more performance in some cases. Specially when using simple arithmetic kernels on the same data, one after the other.

The first main focus is on the transform operation, with an incomplete set of basic arithmetic operations to be performed on cv::cuda::GpuMat objects.

In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc and at least C++14 support.
