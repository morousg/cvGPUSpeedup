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

#pragma once

#include "cv2cuda_types.h"

#include <opencv2/core/cuda.hpp>

template <int I, typename Operator, typename Enabler = void>
struct split_t {};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data};
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data,
                (BASE_CUDA_T(I)*)output.at(2).data};
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data,
                (BASE_CUDA_T(I)*)output.at(2).data, (BASE_CUDA_T(I)*)output.at(3).data};
    }
};

template <int I, typename Operator, typename Enabler = void>
struct operate_t {};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 1>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { static_cast<BASE_CUDA_T(I)>(val[0]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { make_<CUDA_T(I)>(val[0], val[1]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { make_<CUDA_T(I)>(val[0], val[1], val[2]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { make_<CUDA_T(I)>(val[0], val[1], val[2], val[3]) };
    }
};
