/* Copyright 2023 Oscar Amoros Huguet
   Copyright 2023 David Del Rio Astorga

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

    template <typename... Types>
    struct TypeList {};

    template<typename... Args1, typename... Args2, typename... Args3, typename... Args4>
    struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>, TypeList<Args4...>> {
        using type = TypeList<Args1..., Args2..., Args3..., Args4...>;
    };

    using VOne   = TypeList<uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo   = TypeList<uchar2, char2, ushort2, short2, uint2, int2, ulong2, long2, ulonglong2, longlong2, float2, double2>;
    using VThree = TypeList<uchar3, char3, ushort3, short3, uint3, int3, ulong3, long3, ulonglong3, longlong3, float3, double3>;
    using VFour  = TypeList<uchar4, char4, ushort4, short4, uint4, int4, ulong4, long4, ulonglong4, longlong4, float4, double4>;
    using VAll   = typename TypeList<VOne, VTwo, VThree, VFour>::type;

    template <typename... Args>
    struct one_of_t {};

    template <typename T, typename... U>
    struct one_of_t<T, TypeList<U...>> {
        enum {value = (std::is_same<T, U>::value || ...)};
    };

    template <typename T>
    constexpr bool validCUDAVec = one_of_t<T, VAll>::value;

    template <typename T>
    __host__ __device__ __forceinline__ constexpr int Channels() { 
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: Channels<invalid_type>()");
        if constexpr (one_of_t<T, VOne>::value) {
            return 1;
        } else if constexpr (one_of_t<T, VTwo>::value) { 
            return 2;
        } else if constexpr (one_of_t<T, VThree>::value) { 
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
            static_assert(Channels<T>() >= 2, "Vector type smaller than 2 elements has no member y"); 
            return vector.y;
        } else if constexpr (idx == 2) { 
            static_assert(Channels<T>() >= 3, "Vector type smaller than 3 elements has no member z");
            return vector.z;
        } else if constexpr (idx == 3) {
            static_assert(Channels<T>() == 4, "Vector type smaller than 4 elements has no member w");
            return vector.w;
        }
    }

    template <int... idx>
    struct VReorder {
        template <typename T>
        FK_HOST_DEVICE_FUSE T exec(const T& vector) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VReorder<...>::exec<invalid_type>(invalid_type vector)");
            static_assert(sizeof...(idx) == Channels<T>(), "Wrong number of indexes for the cuda vetor type in VReorder.");
            return {VectorAt<idx>(vector)...};
        }
    };

    template <typename V>
    struct VectorTraits {};

#define VECTOR_TRAITS(BaseType) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum {cn=1}; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 1> { using base = BaseType; enum {cn=1}; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 2> { using base = BaseType; enum {cn=2}; enum {bytes=sizeof(base)*2}; }; \
    template <> \
    struct VectorTraits<BaseType ## 3> { using base = BaseType; enum {cn=3}; enum {bytes=sizeof(base)*3}; }; \
    template <> \
    struct VectorTraits<BaseType ## 4> { using base = BaseType; enum {cn=4}; enum {bytes=sizeof(base)*4}; };

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
    

    template <typename T, typename Enabler=void>
    struct to_printable;

    template <typename T>
    struct to_printable<T, std::enable_if_t<sizeof(T) == 1>> {
        FK_HOST_FUSE int exec(T val) { return static_cast<int>(val); }
    };

    template <typename T>
    struct to_printable<T, std::enable_if_t<(sizeof(T) > 1)>> {
        FK_HOST_FUSE T exec(T val) { return val; }
    };

    template <typename T, typename Enabler=void>
    struct print_vector;

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 1>> {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 2>> {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << ", " << to_printable<decltype(T::y)>()(val.y) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 3>> {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
            outs << "{" << to_printable<decltype(T::x)>()(val.x) << ", " << to_printable<decltype(T::y)>()(val.y) <<
            ", " << to_printable<decltype(T::z)>()(val.z) << "}";
            return outs;
        }
    };

    template <typename T>
    struct print_vector<T, typename std::enable_if_t<CN(T) == 4>> {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
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
    struct unary_vector_set_<T, typename std::enable_if_t<!std::is_aggregate<T>::value &&
                                                          !std::is_class<T>::value &&
                                                          !std::is_enum<T>::value>>{
        // This case exists to make things easyer when we don't know if the type
        // is going to be a vector type or a normal type
        FK_HOST_DEVICE_FUSE T exec(const T& val) {
            return val;
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<std::is_aggregate<T>::value &&
                                                          VectorTraits<T>::cn == 1>> {
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
                return {val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<VectorTraits<T>::cn == 2>> {
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
                return {val, val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<VectorTraits<T>::cn == 3>>{
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
                return {val, val, val};
        }
    };

    template <typename T>
    struct unary_vector_set_<T, typename std::enable_if_t<VectorTraits<T>::cn == 4>>{
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
                return {val, val, val, val};
        }
    };

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const typename VectorTraits<T>::base& val) {
        return unary_vector_set_<T>::exec(val);
    }

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const T& val) {
        return unary_vector_set_<T>::exec(val);
    }
}