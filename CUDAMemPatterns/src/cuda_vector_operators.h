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
#include "operation_types.h"

template <typename Operation, typename T, typename Enabler=void>
struct operation_;

template <typename Operation, typename T>
struct operation_<Operation, T, typename std::enable_if_t<NUM_COMPONENTS(T) == 2>> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(Operation()(n1.x, n2.x), Operation()(n1.y, n2.y));
    }
};

template <typename Operation, typename T>
struct operation_<Operation, T, typename std::enable_if_t<NUM_COMPONENTS(T) == 3>> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(Operation()(n1.x, n2.x), Operation()(n1.y, n2.y), Operation()(n1.z, n2.z));
    }
};

template <typename Operation, typename T>
struct operation_<Operation, T, typename std::enable_if_t<NUM_COMPONENTS(T) == 4>> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(Operation()(n1.x, n2.x), Operation()(n1.y, n2.y), Operation()(n1.z, n2.z), Operation()(n1.w, n2.w));
    }
};

template <typename T>
__device__ __host__ T operator-(const T& n1, const T& n2) {
    return operation_<binary_sub<decltype(T::x)>, const T>()(n1, n2);
}

template <typename T>
__device__ __host__ T operator+(const T& n1, const T& n2) {
    return operation_<binary_sum<decltype(T::x)>, const T>()(n1, n2);
}

template <typename T>
__device__ __host__ T operator*(const T& n1, const T& n2) {
    return operation_<binary_mul<decltype(T::x)>, const T>()(n1, n2);
}

template <typename T>
__device__ __host__ T operator/(const T& n1, const T& n2) {
    return operation_<binary_div<decltype(T::x)>, const T>()(n1, n2);
}
