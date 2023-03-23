#pragma once
#include "operation_types.h"

template <typename operation, typename T, typename Enabler=void>
struct operation_;

template <typename operation, typename T>
struct operation_<operation, T, typename std::enable_if<NUM_COMPONENTS(T) == 2>::type> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(operation()(n1.x, n2.x), operation()(n1.y, n2.y));
    }
};

template <typename operation, typename T>
struct operation_<operation, T, typename std::enable_if<NUM_COMPONENTS(T) == 3>::type> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(operation()(n1.x, n2.x), operation()(n1.y, n2.y), operation()(n1.z, n2.z));
    }
};

template <typename operation, typename T>
struct operation_<operation, T, typename std::enable_if<NUM_COMPONENTS(T) == 4>::type> {
    inline constexpr __device__ __host__ T operator()(T& n1, T& n2) {
        return make_<T>(operation()(n1.x, n2.x), operation()(n1.y, n2.y), operation()(n1.z, n2.z), operation()(n1.w, n2.w));
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