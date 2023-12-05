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

#include <fused_kernel/core/utils/cuda_vector_utils.cuh>
#include <fused_kernel/core/fusionable_operations/memory_operations.cuh>
#include <cv2cuda_types.cuh>

#include <opencv2/core/cuda.hpp>

namespace cvGS {
    template <int T>
    inline cv::Scalar cvScalar_set(const BASE_CUDA_T(T)& value) {
        if constexpr (CV_MAT_CN(T) == 1) {
            return cv::Scalar(value);
        }
        else if constexpr (CV_MAT_CN(T) == 2) {
            return cv::Scalar(value, value);
        }
        else if constexpr (CV_MAT_CN(T) == 3) {
            return cv::Scalar(value, value, value);
        }
        else if constexpr (CV_MAT_CN(T) == 4) {
            return cv::Scalar(value, value, value, value);
        }
    }
    template <int I>
    struct cvScalar2CUDAV {
        FK_HOST_FUSE CUDA_T(I) get(const cv::Scalar& val) {
            if constexpr (CV_MAT_CN(I) == 1) {
                return static_cast<BASE_CUDA_T(I)>(val[0]);
            }
            else if constexpr (CV_MAT_CN(I) == 2) {
                return fk::make::type<CUDA_T(I)>(val[0], val[1]);
            }
            else if constexpr (CV_MAT_CN(I) == 3) {
                return fk::make::type<CUDA_T(I)>(val[0], val[1], val[2]);
            }
            else {
                return fk::make::type<CUDA_T(I)>(val[0], val[1], val[2], val[3]);
            }
        }
    };
namespace internal {

    template <int I, typename PtrType, typename Operator>
    struct split_builder_t {
        FK_HOST_FUSE auto build(const std::vector<PtrType>& output) {
            static_assert(CV_MAT_CN(I) >= 2, "Split operations can only be used with types of 2, 3 or 4 channels.");
            if constexpr (CV_MAT_CN(I) == 2) {
                return Operator{ {output.at(0), output.at(1)} };
            } else if constexpr (CV_MAT_CN(I) == 3) {
                return Operator{ {output.at(0), output.at(1), output.at(2)} };
            } else {
                return Operator{ {output.at(0), output.at(1), output.at(2), output.at(3)} };
            }
        }
    };

} // namespace internal
} // namespace cvGS
