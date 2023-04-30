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

#include <fast_kernel/cuda_vector_utils.cuh>
#include <fast_kernel/memory_operation_types.cuh>
#include <cv2cuda_types.h>

#include <opencv2/core/cuda.hpp>

namespace cvGS {

namespace internal {

template <int I, typename PtrType, typename Operator, typename Enabler = void>
struct split_builder_t {};

template <int I, typename PtrType, typename Operator>
struct split_builder_t<I, PtrType, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    FK_HOST_FUSE Operator build(const std::vector<PtrType>& output) {
        return { output.at(0), output.at(1) };
    }
};

template <int I, typename PtrType, typename Operator>
struct split_builder_t<I, PtrType, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    FK_HOST_FUSE Operator build(const std::vector<PtrType>& output) {
        return { output.at(0), output.at(1), output.at(2) };
    }
};

template <int I, typename PtrType, typename Operator>
struct split_builder_t<I, PtrType, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    FK_HOST_FUSE Operator build(const std::vector<PtrType>& output) {
        return { output.at(0), output.at(1), output.at(2), output.at(3) };
    }
};

template <int I, typename Operator, typename Enabler = void>
struct operator_builder_t {};

template <int I, typename Operator>
struct operator_builder_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 1>> {
    FK_HOST_FUSE Operator build(const cv::Scalar& val) {
        return { static_cast<BASE_CUDA_T(I)>(val[0]) };
    }
};

template <int I, typename Operator>
struct operator_builder_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    FK_HOST_FUSE Operator build(const cv::Scalar& val) {
        return { fk::make::type<CUDA_T(I)>(val[0], val[1]) };
    }
};

template <int I, typename Operator>
struct operator_builder_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    FK_HOST_FUSE Operator build(const cv::Scalar& val) {
        return { fk::make::type<CUDA_T(I)>(val[0], val[1], val[2]) };
    }
};

template <int I, typename Operator>
struct operator_builder_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    FK_HOST_FUSE Operator build(const cv::Scalar& val) {
        return { fk::make::type<CUDA_T(I)>(val[0], val[1], val[2], val[3]) };
    }
};
} // namespace internal
} // namespace cvGS
