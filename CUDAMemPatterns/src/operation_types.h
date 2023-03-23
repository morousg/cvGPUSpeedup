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

template <typename I, typename O>
struct unary_cuda_vector_cast {
  __device__ __host__ O operator()(I input) {
      if constexpr (NUM_COMPONENTS(I) == 1) {
        return make_<O>(input.x); 
      } else if constexpr (NUM_COMPONENTS(I) == 2) {
        return make_<O>(input.x, input.y);
      } else if constexpr (NUM_COMPONENTS(I) == 3) {
        return make_<O>(input.x, input.y, input.z);
      } else if constexpr (NUM_COMPONENTS(I) == 4) {
        return make_<O>(input.x, input.y, input.z, input.w);
      } else {
        return 0;
      }
  }
};