#pragma once

#include "cuda_utils.h"

#define NUM_COMPONENTS(some_type) (sizeof(some_type)/sizeof(decltype(some_type::x)))

// Automagically making any CUDA vector type from a template type
// It will not compile if you try to do bad things. The number of elements
// need to conform to T, and the type of the elements will always be casted.
template <typename T, typename... numbers>
inline constexpr __device__ __host__ T make_(numbers... pack) {
    return {static_cast<decltype(T::x)>(pack)...};
}