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
#include "cuda_vector_utils.cuh"
#include "../external/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp"

#define DECL_TYPES_UNARY(I, O) using InputType = I; using OutputType = O;
#define DECL_TYPES_BINARY(I1, I2, O)  using InputType = I1; using ParamsType = I2; using OutputType = O;

namespace fk {

template <typename I1, typename I2=I1, typename O=I1>
struct BinarySum {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 + input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinarySub {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 - input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinaryMul {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) { return input_1 * input_2; }
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinaryDiv {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 / input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename T, int... idxs>
struct UnaryVectorReorder {
    FK_HOST_DEVICE_FUSE T exec(const T& input) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(Channels<T>() >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        return VReorder<idxs...>::exec(input);
    }
    DECL_TYPES_UNARY(T, T)
};

template <typename T>
struct BinaryVectorReorder {
    FK_HOST_DEVICE_FUSE T exec(const T& input, const typename VectorType<int,Channels<T>()>::type& idx) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(Channels<T>() >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        using baseType = typename VectorTraits<T>::base;
        const baseType* const temp = (baseType*)&input; 
        if constexpr (Channels<T>() == 2) {
            return {temp[idx.x], temp[idx.y]};
        } else if constexpr (Channels<T>() == 3) {
            return {temp[idx.x], temp[idx.y], temp[idx.z]};
        } else {
            return {temp[idx.x], temp[idx.y], temp[idx.z], temp[idx.w]};
        }
    }
    using idxType = typename VectorType<int,Channels<T>()>::type;
    DECL_TYPES_BINARY(T, idxType, T)
};

template <typename I, typename O>
struct UnaryCast {
    FK_HOST_DEVICE_FUSE O exec(const I& input) { return saturate_cast<O>(input); }
    DECL_TYPES_UNARY(I, O)
};

template <typename I, typename O>
struct UnaryDiscard {
    FK_HOST_DEVICE_FUSE O exec(const I& input) {
        static_assert(cn<I> > cn<O>, "Output type should at least have one channel less");
        static_assert(std::is_same_v<typename VectorTraits<I>::base,
                                     typename VectorTraits<O>::base>,
                                     "Base types should be the same");
        if constexpr (cn<O> == 1) {
            if constexpr (std::is_aggregate_v<O>) {
                return {input.x};
            } else {
                return input.x;
            }
        } else if constexpr (cn<O> == 2) {
            return { input.x, input.y };
        } else if constexpr (cn<O> == 3) {
            return { input.x, input.y, input.z };
        }
    }
    DECL_TYPES_UNARY(I, O)
};

template <typename... OperationTypes>
struct UnaryOperationSequence {
    template <typename Operation>
    FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const Operation::InputType& input) {
        return Operation::exec(input);
    }
    template <typename Operation, typename... RemainingOperations>
    FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const Operation::InputType& input) {
        return UnaryOperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
    }
    FK_HOST_DEVICE_FUSE typename LastType_t<OperationTypes...>::OutputType exec(const typename FirstType_t<OperationTypes...>::InputType& input) {
        return UnaryOperationSequence<OperationTypes...>::next_exec<OperationTypes...>(input);
    }
    DECL_TYPES_UNARY(typename FirstType_t<OperationTypes...>::InputType, typename LastType_t<OperationTypes...>::OutputType)
};

}

#undef DECL_TYPES_UNARY
#undef DECL_TYPES_BINARY
