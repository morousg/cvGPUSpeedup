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

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/types.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

namespace cvGS {

    template <int type>
    struct cv2cuda_t;

#define CV2CUDA_T(cvType, cuType) \
    template <>                   \
    struct cv2cuda_t<cvType> {    \
        using type = cuType;      \
    };

    CV2CUDA_T(CV_8UC1, uchar)
    CV2CUDA_T(CV_8UC2, uchar2)
    CV2CUDA_T(CV_8UC3, uchar3)
    CV2CUDA_T(CV_8UC4, uchar4)
    CV2CUDA_T(CV_8SC1, char)
    CV2CUDA_T(CV_8SC2, char2)
    CV2CUDA_T(CV_8SC3, char3)
    CV2CUDA_T(CV_8SC4, char4)
    CV2CUDA_T(CV_16UC1, ushort)
    CV2CUDA_T(CV_16UC2, ushort2)
    CV2CUDA_T(CV_16UC3, ushort3)
    CV2CUDA_T(CV_16UC4, ushort4)
    CV2CUDA_T(CV_16SC1, short)
    CV2CUDA_T(CV_16SC2, short2)
    CV2CUDA_T(CV_16SC3, short3)
    CV2CUDA_T(CV_16SC4, short4)
    CV2CUDA_T(CV_32SC1, int)
    CV2CUDA_T(CV_32SC2, int2)
    CV2CUDA_T(CV_32SC3, int3)
    CV2CUDA_T(CV_32SC4, int4)
    CV2CUDA_T(CV_32FC1, float)
    CV2CUDA_T(CV_32FC2, float2)
    CV2CUDA_T(CV_32FC3, float3)
    CV2CUDA_T(CV_32FC4, float4)
    CV2CUDA_T(CV_64FC1, double)
    CV2CUDA_T(CV_64FC2, double2)
    CV2CUDA_T(CV_64FC3, double3)
    CV2CUDA_T(CV_64FC4, double4)

#undef CV2CUDA_T

    template <int... codes>
    struct CodesList {};

    template <int code, typename Codes>
    struct one_of_c : std::false_type {};

    template <int code, int... codes>
    struct one_of_c<code, CodesList<code, codes...>> : std::true_type {};

    template <int code, int otherCode, int... codes>
    struct one_of_c<code, CodesList<otherCode, codes...>> : one_of_c<code, CodesList<codes...>> {};

    using SupportedColorConversions = CodesList<cv::COLOR_BGR2RGB, cv::COLOR_RGB2BGR, cv::COLOR_BGRA2RGBA, cv::COLOR_RGBA2BGRA>;
    using SupportedInterpolations = CodesList<cv::INTER_LINEAR>;

    template <int code>
    constexpr bool isSupportedColorConversion = one_of_c<code, SupportedColorConversions>::value;
    template <int code>
    constexpr bool isSupportedInterpolation = one_of_c<code, SupportedInterpolations>::value;

}

#define CUDA_T(CV_TYPE) typename cvGS::cv2cuda_t<CV_TYPE>::type
#define BASE_CUDA_T(CV_TYPE) typename cvGS::cv2cuda_t<CV_MAT_DEPTH(CV_TYPE)>::type
