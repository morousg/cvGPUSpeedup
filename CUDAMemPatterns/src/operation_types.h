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
struct unary_cuda_vector_cast<I, O, typename std::enable_if<NUM_COMPONENTS(I) == 1>::type> {
  __device__ __host__ O operator()(I input) { return make_<O>(input.x); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if<NUM_COMPONENTS(I) == 2>::type> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if<NUM_COMPONENTS(I) == 3>::type> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y, input.z); }
};

template <typename I, typename O>
struct unary_cuda_vector_cast<I, O, typename std::enable_if<NUM_COMPONENTS(I) == 4>::type> {
    __device__ __host__ O operator()(I input) { return make_<O>(input.x, input.y, input.z, input.w); }
};