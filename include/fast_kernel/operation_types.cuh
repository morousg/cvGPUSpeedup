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
#include "cuda_vector_utils.cuh"
#include "../external/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp"

namespace fk {

template <typename I1, typename I2=I1, typename O=I1>
struct binary_sum {
    __device__ __forceinline__ __host__ constexpr O operator()(I1 input_1, I2 input_2) {return input_1 + input_2;}
};

template <typename I1, typename I2=I1, typename O=I1, typename Enabler=void>
struct binary_sub {
    __device__ __forceinline__ __host__ constexpr O operator()(I1 input_1, I2 input_2) {return input_1 - input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_mul {
    __device__ __forceinline__ __host__ constexpr O operator()(I1 input_1, I2 input_2) { return input_1 * input_2; }
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_div {
    __device__ __forceinline__ __host__ constexpr O operator()(I1 input_1, I2 input_2) {return input_1 / input_2;}
};

template <typename I, typename O>
struct unary_cast {
    __device__ __forceinline__  __host__ constexpr O operator()(I input) { return saturate_cast<O>(input); }
};

}