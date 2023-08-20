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
#include "type_lists.cuh"

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

    template <typename BaseType, int Channels>
    using VectorType_t = typename VectorType<BaseType, Channels>::type;

    using VOne   = TypeList<uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo   = TypeList<uchar2, char2, ushort2, short2, uint2, int2, ulong2, long2, ulonglong2, longlong2, float2, double2>;
    using VThree = TypeList<uchar3, char3, ushort3, short3, uint3, int3, ulong3, long3, ulonglong3, longlong3, float3, double3>;
    using VFour  = TypeList<uchar4, char4, ushort4, short4, uint4, int4, ulong4, long4, ulonglong4, longlong4, float4, double4>;
    using VAll   = typename TypeList<VOne, VTwo, VThree, VFour>::type;

    template <typename T>
    constexpr bool validCUDAVec = one_of<T, VAll>::value;

    template <typename T>
    __host__ __device__ __forceinline__ constexpr int Channels() {
        if constexpr (one_of_v<T, VOne> || !validCUDAVec<T>) {
            return 1;
        } else if constexpr (one_of_v<T, VTwo>) { 
            return 2;
        } else if constexpr (one_of_v<T, VThree>) { 
            return 3;
        } else {
            return 4;
        }
    }

    template <typename T>
    constexpr int cn = Channels<T>();

    template <int idx, typename T>
    __host__ __device__ __forceinline__ constexpr auto VectorAt(const T& vector) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VectorAt<invalid_type>()");
        if constexpr (idx == 0) { 
            return vector.x; 
        } else if constexpr (idx == 1) { 
            static_assert(cn<T> >= 2, "Vector type smaller than 2 elements has no member y"); 
            return vector.y;
        } else if constexpr (idx == 2) { 
            static_assert(cn<T> >= 3, "Vector type smaller than 3 elements has no member z");
            return vector.z;
        } else if constexpr (idx == 3) {
            static_assert(cn<T> == 4, "Vector type smaller than 4 elements has no member w");
            return vector.w;
        }
    }

    template <int... idx>
    struct VReorder {
        template <typename T>
        FK_HOST_DEVICE_FUSE T exec(const T& vector) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VReorder<...>::exec<invalid_type>(invalid_type vector)");
            static_assert(sizeof...(idx) == cn<T>, "Wrong number of indexes for the cuda vetor type in VReorder.");
            return {VectorAt<idx>(vector)...};
        }
    };

    template <typename V>
    struct VectorTraits {};

#define VECTOR_TRAITS(BaseType) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 1> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 2> { using base = BaseType; enum {bytes=sizeof(base)*2}; }; \
    template <> \
    struct VectorTraits<BaseType ## 3> { using base = BaseType; enum {bytes=sizeof(base)*3}; }; \
    template <> \
    struct VectorTraits<BaseType ## 4> { using base = BaseType; enum {bytes=sizeof(base)*4}; };

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
    struct make {
        template <typename T, typename... Numbers>
        FK_HOST_DEVICE_FUSE T type(const Numbers&... pack) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: make::type<invalid_type>()");
            return {static_cast<decltype(T::x)>(pack)...};
        }
    };

    template <typename T, typename... Numbers>
    FK_HOST_DEVICE_CNST T make_(const Numbers&... pack) {
        return make::type<T>(pack...);
    }
    
    template <typename T, typename Enabler=void>
    struct UnaryVectorSet;
    
    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<!validCUDAVec<T>>>{
        // This case exists to make things easier when we don't know if the type
        // is going to be a vector type or a normal type
        FK_HOST_DEVICE_FUSE T exec(const T& val) {
            return val;
        }
    };

    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<validCUDAVec<T>>> {
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
            if constexpr (cn<T> == 1) {
                return {val};
            } else if constexpr (cn<T> == 2) {
                return {val, val};
            } else if constexpr (cn<T> == 3) {
                return {val, val, val};
            } else {
                return {val, val, val, val};
            }
        }
    };

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const typename VectorTraits<T>::base& val) {
        return UnaryVectorSet<T>::exec(val);
    }

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const T& val) {
        return UnaryVectorSet<T>::exec(val);
    }

    template <typename T>
    struct to_printable {
        FK_HOST_FUSE int exec(T val) {
            if constexpr (sizeof(T) == 1) {
                return static_cast<int>(val);
            } else if constexpr (sizeof(T) > 1) {
                return val;
            }
        }
    };

    template <typename T>
    struct print_vector {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
            if constexpr (!validCUDAVec<T>) {
                outs << val;
                return outs;
            } else if constexpr (cn<T> == 1) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) << "}";
                return outs;
            } else if constexpr (cn<T> == 2) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                       ", " << to_printable<decltype(T::y)>::exec(val.y) << "}";
                return outs;
            } else if constexpr (cn<T> == 3) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                       ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                       ", " << to_printable<decltype(T::z)>::exec(val.z) << "}";
                return outs;
            } else {
                 outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                        ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                        ", " << to_printable<decltype(T::z)>::exec(val.z) <<
                        ", " << to_printable<decltype(T::w)>::exec(val.w) << "}";
                return outs;
            }
        }
    };

    template <typename T> 
    __host__ inline constexpr typename std::enable_if_t<validCUDAVec<T>, std::ostream&> operator<<(std::ostream& outs, const T& val) {
        return print_vector<T>::exec(outs, val);
    }
}
