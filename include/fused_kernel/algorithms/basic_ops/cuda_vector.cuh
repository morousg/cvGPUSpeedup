/* Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_CUDA_VECTOR
#define FK_CUDA_VECTOR

#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/logical.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    template <typename I, typename O>
    struct Discard {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
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
        }
        using InstantiableType = Unary<Discard<I, O>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename T, int... Idx>
    struct VReorder {
        using InputType = T;
        using OutputType = T;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE T exec(const T& vector) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VReorder<...>::exec<invalid_type>(invalid_type vector)");
            static_assert(sizeof...(Idx) == cn<T>, "Wrong number of indexes for the cuda vetor type in VReorder.");
            return { VectorAt<Idx>(vector)... };
        }
        using InstantiableType = Unary<VReorder<T, Idx...>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename T, int... Idx>
    struct VectorReorder {
        using InputType = T;
        using OutputType = T;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnaryVectorReorder");
            static_assert(cn<InputType> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
            return VReorder<T, Idx...>::exec(input);
        }
        using InstantiableType = Unary<VectorReorder<T, Idx...>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename T, typename Operation>
    struct VectorReduce { 
        using InputType = T;
        using OutputType = VBase<T>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                if constexpr (cn<T> == 1) {
                    if constexpr (validCUDAVec<T>) {
                        return input.x;
                    } else {
                        return input;
                    }
                } else if constexpr (cn<T> == 2) {
                    return Operation::exec({ input.x, input.y });
                } else if constexpr (cn<T> == 3) {
                    return Operation::exec({ Operation::exec({ input.x, input.y }), input.z });
                } else if constexpr (cn<T> == 4) {
                    return Operation::exec({ Operation::exec({ Operation::exec({ input.x, input.y }), input.z }), input.w });
                }
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                if constexpr (cn<T> == 1) {
                    if constexpr (validCUDAVec<T>) {
                        return input.x;
                    } else {
                        return input;
                    }
                } else if constexpr (cn<T> == 2) {
                    return Operation::exec(input.x, input.y);
                } else if constexpr (cn<T> == 3) {
                    return Operation::exec(Operation::exec(input.x, input.y), input.z);
                } else if constexpr (cn<T> == 4) {
                    return Operation::exec(Operation::exec(Operation::exec(input.x, input.y), input.z), input.w);
                }
            }
        }
        using InstantiableType = Unary<VectorReduce<T, Operation>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename I, typename O>
    struct AddLast {
        using InputType = I;
        using OutputType = O;
        using ParamsType = typename VectorTraits<I>::base;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<AddLast<I, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            static_assert(cn<InputType> == cn<OutputType> -1, "Output type should have one channel more");
            static_assert(std::is_same_v<typename VectorTraits<InputType>::base, typename VectorTraits<OutputType>::base>,
                "Base types should be the same");
            const ParamsType newElem = opData.params;
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
        using InstantiableType = Binary<AddLast<I, O>>;
        DEFAULT_BUILD
    };

    template <typename T>
    struct VectorAnd {
        static_assert(std::is_same_v<VBase<T>, bool>, "VectorAnd only works with boolean vectors");
        using InputType = T;
        using OutputType = bool;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return VectorReduce<T, Equal<bool, bool>>::exec(input);
        }
        using InstantiableType = Unary<VectorAnd<T>>;
        DEFAULT_UNARY_BUILD
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
