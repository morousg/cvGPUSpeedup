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

bool checkResults(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& h_comparison1C) {
    cv::Mat h_comparison;
    cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32F, 0.0001f);
    cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_LT);
    
    int errors = cv::countNonZero(h_comparison1C);
    return errors == 0;
}

void testSplitOutputOperation(const cv::cuda::GpuMat& d_input, int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream) {
    cv::cuda::GpuMat d_temp(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32FC3);
    cv::cuda::GpuMat d_temp2(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32FC3);

    std::vector<cv::cuda::GpuMat> d_output_cv(3);
    for (auto& mat : d_output_cv) {
        mat.create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32F);
    }
    
    std::vector<cv::cuda::GpuMat> d_output_cvGS(3);
    for (auto& mat : d_output_cvGS) {
        mat.create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_32F);
    }
    double alpha = 0.3;
    cv::Scalar val_sub(1.f, 4.f, 3.2f);
    cv::Scalar val_div(3.2f, 0.6f, 11.8f);

    // OpenCV version
    d_input.convertTo(d_temp, CV_32FC3, alpha, cv_stream);
    cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
    cv::cuda::divide(d_temp2, val_div, d_temp, 1.0, -1, cv_stream);
    cv::cuda::split(d_temp, d_output_cv, cv_stream);

    // cvGPUSpeedup version
    cvGS::executeOperations<CV_8UC3>(d_input, cv_stream, 
                                               cvGS::convertTo<CV_8UC3, CV_32FC3>(),
                                               cvGS::multiply<CV_32FC3>(cv::Scalar(alpha, alpha, alpha)),
                                               cvGS::subtract<CV_32FC3>(val_sub),
                                               cvGS::divide<CV_32FC3>(val_div),
                                               cvGS::split<CV_32FC3>(d_output_cvGS));

    // Looking at Nsight Systems, with an RTX A2000 12GB
    // OpenCV version execution time CPU (only launching the kernels) + GPU kernels only = 36600us
    // cvGS version execution time CPU (only launching the kernels) + GPU kernels only = 807us
    // Speed up = 45.35x
    // OpenCV version execution time GPU kernels only = 228us + 299us + 298us + 352us = 1177us
    // cvGS version execution time GPU kernels only = 124us
    // Speed up = 9.5x

    // Verify results
    std::vector<cv::Mat> h_cvResults(3);
    std::vector<cv::Mat> h_cvGSResults(3);

    for (int i=0; i<3; i++) {
        d_output_cv.at(i).download(h_cvResults.at(i), cv_stream);
        d_output_cvGS.at(i).download(h_cvGSResults.at(i), cv_stream);
    }

    cv_stream.waitForCompletion();

    bool passed = true;
    for (int i=0; i<3; i++) {
        cv::Mat diff = cv::abs(h_cvResults.at(i) - h_cvGSResults.at(i));
        passed &= checkResults(NUM_ELEMS_X, NUM_ELEMS_Y, diff);
    }

    if (passed) {
        std::cout << "testSplitOutputOperation passed!!" << std::endl;
    } else {
        std::cout << "testSplitOutputOperation failed!!" << std::endl;
    }
}

void testNoDefinedOutputOperation(const cv::cuda::GpuMat& d_input, int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream) {
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

    d_output_cv.download(h_cvResults, cv_stream);
    d_output_cvGS.download(h_cvGSResults, cv_stream);

    cv_stream.waitForCompletion();

    cv::Mat diff = cv::abs(h_cvResults - h_cvGSResults);
    cv::Mat h_comparison1C;
    cv::cvtColor(diff, h_comparison1C, cv::COLOR_RGB2GRAY, 1);

    if (checkResults(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C)) {
        std::cout << "testNoDefinedOutputOperation passed!!" << std::endl;
    } else {
        std::cout << "testNoDefinedOutputOperation failed!!" << std::endl;
    }
}

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

    testNoDefinedOutputOperation(d_input, NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream);
    testSplitOutputOperation(d_input, NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream);
    
    return 0;
}