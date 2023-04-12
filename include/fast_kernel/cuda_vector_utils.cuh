/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include "cuda_utils.cuh"

#define CN(v_type) sizeof(v_type) / sizeof(decltype(v_type::x))

namespace fk {

    template <typename BaseType, int Channels>
    struct VectorType {};

#define VECTOR_TYPE(BaseType) \
    template <> \
    struct VectorType<BaseType, 1> { using type = BaseType; }; \
    template <> \
    struct VectorType<BaseType, 2> { using type = BaseType ## 2; }; \
    template <> \
    struct VectorType<BaseType, 3> { using type = BaseType ## 3; }; \
    template <> \
    struct VectorType<BaseType, 4> { using type = BaseType ## 4; };

    VECTOR_TYPE(uchar)
    VECTOR_TYPE(char)
    VECTOR_TYPE(short)
    VECTOR_TYPE(ushort)
    VECTOR_TYPE(int)
    VECTOR_TYPE(uint)
    VECTOR_TYPE(long)
    VECTOR_TYPE(ulong)
    VECTOR_TYPE(longlong)
    VECTOR_TYPE(ulonglong)
    VECTOR_TYPE(float)
    VECTOR_TYPE(double)
#undef VECTOR_TYPE

    template <typename V>
    struct VectorTraits {};

#define VECTOR_TRAITS(BaseType) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum {cn=1}; }; \
    template <> \
    struct VectorTraits<BaseType ## 1> { using base = BaseType; enum {cn=1}; }; \
    template <> \
    struct VectorTraits<BaseType ## 2> { using base = BaseType; enum {cn=2}; }; \
    template <> \
    struct VectorTraits<BaseType ## 3> { using base = BaseType; enum {cn=3}; }; \
    template <> \
    struct VectorTraits<BaseType ## 4> { using base = BaseType; enum {cn=4}; };

    VECTOR_TRAITS(uchar)
    VECTOR_TRAITS(char)
    VECTOR_TRAITS(short)
    VECTOR_TRAITS(ushort)
    VECTOR_TRAITS(int)
    VECTOR_TRAITS(uint)
    VECTOR_TRAITS(long)
    VECTOR_TRAITS(ulong)
    VECTOR_TRAITS(longlong)
    VECTOR_TRAITS(ulonglong)
    VECTOR_TRAITS(float)
    VECTOR_TRAITS(double)
#undef VECTOR_TRAITS

    // Automagically making any CUDA vector type from a template type
    // It will not compile if you try to do bad things. The number of elements
    // need to conform to T, and the type of the elements will always be casted.
    template <typename T, typename... Numbers>
    __device__ __forceinline__ __host__ constexpr T make_(Numbers... pack) {
        return {static_cast<decltype(T::x)>(pack)...};
    }

    template <typename T, typename Enabler=void>
    struct to_printable;

    template <typename T>
    struct to_printable<T, std::enable_if_t<sizeof(T) == 1>> {
        __host__ inline constexpr int operator()(T val) { return static_cast<int>(val); }
    };

    template <typename T>
    struct to_printable<T, std::enable_if_t<(sizeof(T) > 1)>> {
        __host__ inline constexpr T operator()(T val) { return val; }
    };

    template <typename T, typename Enabler=void>
    struct print_vector;

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 1>> {
        __host__ inline constexpr std::ostream& operator()(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 2>> {
        __host__ inline constexpr std::ostream& operator()(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << ", " << to_printable<decltype(T::y)>()(val.y) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 3>> {
        __host__ inline constexpr std::ostream& operator()(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << ", " << to_printable<decltype(T::y)>()(val.y) <<
            ", " << to_printable<decltype(T::z)>()(val.z) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 4>> {
        __host__ inline constexpr std::ostream& operator()(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << ", " << to_printable<decltype(T::y)>()(val.y) << ", " <<
            to_printable<decltype(T::z)>()(val.z) << ", " << to_printable<decltype(T::w)>()(val.w) << "}";
            return outs;
        }
    };

    template <typename T> 
    __host__ inline constexpr typename std::enable_if_t<std::is_class<T>::value, std::ostream&> operator<<(std::ostream& outs, const T& val) {
        return print_vector<T>()(outs, val);
    }

    template <typename T, typename Enabler=void>
    struct unary_vector_set_;
    
    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<!std::is_class<T>::value>>{
        // This case exists to make things easyer when we don't know if the type
        // is going to be a vector type or a normal type
        __device__ __forceinline__ __host__ T operator()(T& val) {
            return val;
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<std::is_class<T>::value &&
                                                          !std::is_enum<T>::value &&
                                                          CN(T) == 1>>{
        __device__ __forceinline__ __host__ T operator()(decltype(T::x)& val) {
            return {val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<CN(T) == 2>>{
        __device__ __forceinline__ __host__ T operator()(decltype(T::x)& val) {
            return {val, val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<CN(T) == 3>>{
        __device__ __forceinline__ __host__ T operator()(decltype(T::x)& val) {
            return {val, val, val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<CN(T) == 4>>{
        __device__ __forceinline__ __host__ T operator()(decltype(T::x)& val) {
            return {val, val, val, val};
        }
    };

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(decltype(T::x) val) {
        return unary_vector_set_<T>()(val);
    }

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(T val) {
        return unary_vector_set_<T>()(val);
    }
}