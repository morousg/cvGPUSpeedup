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

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include "tests/main.h"

int launch() {

    cv::cuda::Stream cv_stream;

    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(cv_stream); 

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);

    cv::cuda::GpuMat d_input(1080, 1920, CV_32FC3);
    cv::cuda::GpuMat d_output(1080, 1920, CV_32FC3);
    cv::Scalar val(1.f, 4.f, 3.2f);

    cv::cuda::add(d_input, val, d_output, cv::noArray(), -1, cv_stream);

    cv_stream.waitForCompletion();

    std::cout << "OpenCV correctly installed!!" << std::endl;

    return 0;
}