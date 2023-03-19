#pragma once

#include "utils.h"

// Automagically making any CUDA vector type from a template type
// It will not compile if you try to do bad things. The number of elements
// need to conform to T, and the type of the elements will always be casted.
template <typename T, typename... numbers>
T __device__ __host__ make_(numbers... pack) {
    return {static_cast<decltype(T::x)>(pack)...};
}