/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <cvGPUSpeedup.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    // Note that cvGPUSpeedup so far only supports vectors, not matrices
    // since OpenCV will add extra memory for memory alignment reasons,
    // according to cudaMallocPitch https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
    // In the future we will support kernels with step, that can be different for input and output matrices.
    constexpr size_t NUM_ELEMS_X = 1920*1080;
    constexpr size_t NUM_ELEMS_Y = 1;

    cv::cuda::Stream cv_stream;

    cv::Scalar initVal(2u, 37u, 128u);
    cv::Mat h_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3, initVal);

    cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3);
    d_input.upload(h_input, cv_stream);
    cv::cuda::GpuMat d_temp(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32FC3);
    cv::cuda::GpuMat d_output_cv(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32FC3);
    cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32FC3);
    cv::Scalar val_sub(1.f, 4.f, 3.2f);
    cv::Scalar val_mul(3.f, 0.5f, 13.8f);
    cv::Scalar val_div(3.2f, 0.6f, 11.8f);

    // OpenCV version
    d_input.convertTo(d_temp, CV_32FC3, cv_stream);
    cv::cuda::subtract(d_temp, val_sub, d_output_cv, cv::noArray(), -1, cv_stream);
    cv::cuda::multiply(d_output_cv, val_mul, d_temp, 1.0, -1, cv_stream);
    cv::cuda::divide(d_temp, val_div, d_output_cv, 1.0, -1, cv_stream);

    // cvGPUSpeedup version
    cvGS::executeOperations<CV_8UC3, CV_32FC3>(d_input, d_output_cvGS, cv_stream, 
                                               cvGS::convertTo<CV_8UC3, CV_32FC3>(),
                                               cvGS::subtract<CV_32FC3>(val_sub),
                                               cvGS::multiply<CV_32FC3>(val_mul),
                                               cvGS::divide<CV_32FC3>(val_div));

    cv_stream.waitForCompletion();

    // Looking at Nsight Systems, with an RTX A2000 12GB
    // OpenCV version execution time CPU (only launching the kernels) + GPU kernels only = 36600us
    // cvGS version execution time CPU (only launching the kernels) + GPU kernels only = 807us
    // Speed up = 45.35x
    // OpenCV version execution time GPU kernels only = 228us + 299us + 298us + 352us = 1177us
    // cvGS version execution time GPU kernels only = 124us
    // Speed up = 9.5x

    // Verify results
    cv::Mat h_cvResults;
    cv::Mat h_cvGSResults;
    cv::Mat h_comparison;

    d_output_cv.download(h_cvResults);
    d_output_cvGS.download(h_cvGSResults);


    cv::Mat diff = cv::abs(h_cvResults - h_cvGSResults);
    cv::Mat h_comparison1C;
    cv::cvtColor(diff, h_comparison1C, cv::COLOR_RGB2GRAY, 1);
    cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32F, 0.0001f);
    cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_LT);
    
    int errors = cv::countNonZero(h_comparison1C);
    if (errors == 0) {
        std::cout << "Test passed!!" << std::endl;
    } else {
        std::cout << "Test failed!! Number of errors = " << errors << std::endl;
    }

    return 0;
}