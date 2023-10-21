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

#include <sstream>

#include "testsCommon.cuh"
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>

constexpr std::array<size_t, 17> batchValues{ 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80 };

template <int CV_TYPE_I, int CV_TYPE_O, size_t NumOps>
struct VerticalFusion {};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 1> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
                                     const int& BATCH,
                                     const cv::cuda::Stream& cv_stream,
                                     const float& alpha,
                                     const cv::Scalar& val_mul,
                                     const cv::cuda::GpuMat& d_tensor_output,
                                     const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O> 
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 5> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O> 
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 10> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 15> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 20> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 25> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 30> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 35> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 40> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 45> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 50> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 55> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 60> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};
template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 65> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 70> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 75> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusion<CV_TYPE_I, CV_TYPE_O, 80> {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::Scalar& val_mul,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize) {
        cvGS::executeOperations(crops, BATCH, cv_stream,
            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),
            cvGS::multiply<CV_TYPE_O>(val_mul),

            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O, size_t BATCH>
bool benchmark_vertical_fusion(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    constexpr size_t REAL_BATCH{ 50 };
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        struct Parameters {
            const cv::Scalar init;
            const cv::Scalar alpha;
            const cv::Scalar val_sub;
            const cv::Scalar val_div;
        };

        double alpha = 1.0;

        const Parameters one{ {1u}, {alpha}, {1.f}, {3.2f} };
        const Parameters two{ {1u, 2u}, {alpha, alpha}, {1.f, 4.f}, {3.2f, 0.6f} };
        const Parameters three{ {1u, 2u, 3u}, {alpha, alpha, alpha}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f} };
        const Parameters four{ {1u, 2u, 3u, 4u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f} };
        const std::array<Parameters, 4> params{ one, two, three, four };

        const cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
        const cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
        const cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_sub;
        const cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_div;
        try {
            const cv::Size cropSize(60, 120);
            cv::cuda::GpuMat d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
            std::array<cv::cuda::GpuMat, REAL_BATCH> d_output_cv;
            std::array<cv::Mat, REAL_BATCH> h_cvResults;
            std::array<cv::Mat, REAL_BATCH> h_cvGSResults;

            cv::cuda::GpuMat d_temp(cropSize, CV_TYPE_O);
            cv::cuda::GpuMat d_temp2(cropSize, CV_TYPE_O);

            cv::cuda::GpuMat d_tensor_output(REAL_BATCH,
                cropSize.width * cropSize.height,
                CV_TYPE_O);
            d_tensor_output.step = cropSize.width * cropSize.height * sizeof(CUDA_T(CV_TYPE_O));

            cv::Mat diff(cropSize, CV_TYPE_O);
            cv::Mat h_tensor_output(REAL_BATCH, cropSize.width * cropSize.height, CV_TYPE_I);

            std::array<cv::cuda::GpuMat, REAL_BATCH> crops;
            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                crops[crop_i] = cv::cuda::GpuMat(cropSize, CV_TYPE_I, val_init);
                d_output_cv[crop_i].create(cropSize, CV_TYPE_O);
                h_cvResults[crop_i].create(cropSize, CV_TYPE_O);
            }
            
            START_OCV_BENCHMARK
                // OpenCV version
                for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                    crops[crop_i].convertTo(d_output_cv[crop_i], CV_TYPE_O, alpha, cv_stream);
                    for (int numOp = 0; numOp < BATCH; numOp++) {
                        cv::cuda::multiply(d_output_cv[crop_i], val_sub, d_output_cv[crop_i], 1.0, -1, cv_stream);
                    }
                }

            STOP_OCV_START_CVGS_BENCHMARK
                // cvGPUSpeedup
                VerticalFusion<CV_TYPE_I, CV_TYPE_O, BATCH>::execute(crops, REAL_BATCH, cv_stream, alpha, val_sub, d_tensor_output, cropSize);

            STOP_CVGS_BENCHMARK

                d_tensor_output.download(h_tensor_output, cv_stream);

            // Verify results
            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                d_output_cv[crop_i].download(h_cvResults[crop_i], cv_stream);
            }

            cv_stream.waitForCompletion();

            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                cv::Mat cvRes = h_cvResults[crop_i];
                cv::Mat cvGSRes = cv::Mat(cropSize.height, cropSize.width, CV_TYPE_O, h_tensor_output.row(crop_i).data);
                bool passedThisTime = compareAndCheck<CV_TYPE_O>(cropSize.width, cropSize.height, cvRes, cvGSRes);
                if (!passedThisTime) { std::cout << "Failed on crop idx=" << crop_i << std::endl; }
                passed &= passedThisTime;
            }
        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            }
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }
        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "benchmark_vertical_fusion<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "benchmark_vertical_fusion<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool launch_benchmark_vertical_fusion(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
    bool passed = true;

    int dummy[] = { (passed &= benchmark_vertical_fusion<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
    (void)dummy;

    return passed;
}

int main() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["benchmark_vertical_fusion"] = true;
    std::make_index_sequence<batchValues.size()> iSeq{};
#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["benchmark_vertical_fusion"] &= launch_benchmark_vertical_fusion<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream, true);

#ifdef ENABLE_BENCHMARK
    // Warming up for the benchmarks
    results["benchmark_vertical_fusion"] &= benchmark_vertical_fusion<CV_8UC1, CV_32FC1, 5>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);
#endif
    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)

    CLOSE_BENCHMARK

    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
        }
    }

    typename fk::BatchRead<fk::PerThreadRead<fk::_2D, uchar4>, 50>::ParamsType params;

#undef LAUNCH_TESTS

    return 0;
}