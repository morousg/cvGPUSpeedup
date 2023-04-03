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
#include "cuda_vector_utils.h"

template <typename I1, typename I2=I1, typename O=I1>
struct binary_sum {
    inline constexpr __device__ __host__ O operator()(I1 input_1, I2 input_2) {return input_1 + input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_sub {
    inline constexpr __device__ __host__ O operator()(I1 input_1, I2 input_2) {return input_1 - input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_mul {
    inline constexpr __device__ __host__ O operator()(I1 input_1, I2 input_2) {return input_1 * input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_div {
    inline constexpr __device__ __host__ O operator()(I1 input_1, I2 input_2) {return input_1 / input_2;}
};

template <typename I, typename O, typename Enabler=void>
struct unary_cuda_vector_cast {};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if_t<!std::is_class<I>::value &&
                                                              !std::is_class<O>::value >> {
  __device__ __host__ O operator()(I input) { return static_cast<O>(input); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if_t<NUM_COMPONENTS(I) == 1 &&
                                                              std::is_class<I>::value &&
                                                              !std::is_enum<I>::value &&
                                                              std::is_class<O>::value &&
                                                              !std::is_enum<O>::value >> {
  __device__ __host__ O operator()(I input) { return make_<O>(input.x); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if_t<NUM_COMPONENTS(I) == 2>> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if_t<NUM_COMPONENTS(I) == 3>> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y, input.z); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if_t<NUM_COMPONENTS(I) == 4>> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y, input.z, input.w); }
};