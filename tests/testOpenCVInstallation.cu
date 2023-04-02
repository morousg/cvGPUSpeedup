#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

int main() {

    cv::cuda::Stream cv_stream;

    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(cv_stream); 

    cudaStream_t stream;
    error_t err = cudaStreamCreate(&stream);

    cv::cuda::GpuMat d_input(1080, 1920, CV_32FC3);
    cv::cuda::GpuMat d_output(1080, 1920, CV_32FC3);
    cv::Scalar val(1.f, 4.f, 3.2f);

    cv::cuda::add(d_input, val, d_output, cv::noArray(), -1, cv_stream);

    cv_stream.waitForCompletion();

    std::cout << "Test passed!!" << std::endl;

    return 0;
}