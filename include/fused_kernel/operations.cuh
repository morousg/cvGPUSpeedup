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
#include "ptr_nd.cuh"
#include "cuda_vector_utils.cuh"
#include "../external/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp"

namespace fk {

#define Unary(Name) \
template <typename I, typename O> \
struct Unary##Name { \
    using InputType = I; using OutputType = O; \
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
#define FUNCTION_CLOSE }};

Unary(Cast)
    return saturate_cast<O>(input);
FUNCTION_CLOSE

Unary(Discard)
    static_assert(cn<I> > cn<O>, "Output type should at least have one channel less");
    static_assert(std::is_same_v<typename VectorTraits<I>::base,
        typename VectorTraits<O>::base>,
        "Base types should be the same");
    if constexpr (cn<O> == 1) {
        if constexpr (std::is_aggregate_v<O>) {
            return { input.x };
        } else {
            return input.x;
        }
    } else if constexpr (cn<O> == 2) {
        return { input.x, input.y };
    } else if constexpr (cn<O> == 3) {
        return { input.x, input.y, input.z };
    }
FUNCTION_CLOSE
#undef Unary

#define UNARY_DECL_EXEC(I, O) \
using InputType = I; using OutputType = O; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input)

template <typename T, int... idxs>
struct UnaryVectorReorder {
    UNARY_DECL_EXEC(T, T) {
        static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(cn<InputType> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        return VReorder<idxs...>::exec(input);
    }
};

template <typename... OperationTypes>
struct UnaryOperationSequence {
    UNARY_DECL_EXEC(typename FirstType_t<OperationTypes...>::InputType, typename LastType_t<OperationTypes...>::OutputType) {
        return UnaryOperationSequence<OperationTypes...>::next_exec<OperationTypes...>(input);
    }
    template <typename Operation>
    FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const Operation::InputType& input) {
        return Operation::exec(input);
    }
    template <typename Operation, typename... RemainingOperations>
    FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const Operation::InputType& input) {
        return UnaryOperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
    }
};

#undef UNARY_DECL_EXEC

#define Binary(Name) \
template <typename I, typename P=I, typename O=I> \
struct Binary##Name { \
using InputType = I; using ParamsType = P; using OutputType = O; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {

Binary(Sum)
return input + params;
FUNCTION_CLOSE

Binary(Sub)
return input - params;
FUNCTION_CLOSE

Binary(Mul)
return input * params;
FUNCTION_CLOSE

Binary(Div)
return input / params;
FUNCTION_CLOSE

Binary(Max)
return input >= params ? input : params;
FUNCTION_CLOSE

Binary(Min)
return input <= params ? input : params;
FUNCTION_CLOSE
#undef Binary
#undef FUNCTION_CLOSE

#define BINARY_DECL_EXEC(O, I, P) \
using OutputType = O; using InputType = I; using ParamsType = P; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params)

template <typename T>
struct BinaryVectorReorder {
    using BaseType = typename VectorTraits<T>::base;
    using TMP_ParamsType = VectorType_t<int, cn<T>>;
    BINARY_DECL_EXEC(T, T, TMP_ParamsType) {
        static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: BinaryVectorReorder");
        static_assert(cn<InputType> >= 2, "Minimum number of channels is 2: BinaryVectorReorder");
        const BaseType* const temp = (BaseType*)&input;
        const ParamsType idx = params;
        if constexpr (cn<InputType> == 2) {
            return { getValue(idx.x, input), getValue(idx.y, input) };
        } else if constexpr (cn<T> == 3) {
            return { getValue(idx.x, input), getValue(idx.y, input), getValue(idx.z, input) };
        } else {
            return { getValue(idx.x, input), getValue(idx.y, input), getValue(idx.z, input), getValue(idx.w, input) };
        }
    }

    static constexpr __device__ __forceinline__ BaseType getValue(int idx, InputType vector) {
        switch (idx) {
            case 0:
                return vector.x;
            case 1:
                return vector.y;
            case 2:
                return vector.z;
            case 3:
                return vector.w;
        }
    }
};

template <typename I, typename O>
struct BinaryAddLast {
    BINARY_DECL_EXEC(O, I, typename VectorTraits<I>::base) {
        static_assert(cn<InputType> == cn<OutputType> -1, "Output type should have one channel more");
        static_assert(std::is_same_v<typename VectorTraits<InputType>::base, typename VectorTraits<OutputType>::base>,
            "Base types should be the same");
        const ParamsType newElem = params;
        if constexpr (cn<InputType> == 1) {
            if constexpr (std::is_aggregate_v<InputType>) {
                return { input.x, newElem };
            } else {
                return { input, newElem };
            }
        } else if constexpr (cn<InputType> == 2) {
            return { input.x, input.y, newElem };
        } else if constexpr (cn<InputType> == 3) {
            return { input.x, input.y, input.z, newElem };
        }
    }
};

enum InterpolationType {
    // bilinear interpolation
    INTER_LINEAR = 1,
    NONE = 17
};

template <typename I, InterpolationType INTER_T>
struct BinaryInterpolate;

template <typename I>
struct BinaryInterpolate<I, InterpolationType::INTER_LINEAR> {
    using TmpParams = RawPtr<_2D, I>;
    using TmpOutput = VectorType_t<float, cn<I>>;
    BINARY_DECL_EXEC(TmpOutput, float2, TmpParams) {
        const float src_x = input.x;
        const float src_y = input.y;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = BinaryMin<int>::exec(x2, params.dims.width - 1);
        const int y2_read = BinaryMin<int>::exec(y2, params.dims.height - 1);

        OutputType out = make_set<OutputType>(0.f);
        I src_reg = *PtrAccessor<_2D>::cr_point(Point(x1, y1), params);
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));
        src_reg = *PtrAccessor<_2D>::cr_point(Point(x2_read, y1), params);
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));
        src_reg = *PtrAccessor<_2D>::cr_point(Point(x1, y2_read), params);
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));
        src_reg = *PtrAccessor<_2D>::cr_point(Point(x2_read, y2_read), params);
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        return out;
    }
};
#undef BINARY_DECL_EXEC
} //namespace fk
