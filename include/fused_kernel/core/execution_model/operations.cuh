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
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/utils/vlimits.cuh>
#include <fused_kernel/core/utils/operation_tuple.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>

#include <climits>

namespace fk {

#define UNARY_DECL_EXEC(I, O) \
using InputType = I; using OutputType = O; using InstanceType = UnaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input)

#define BINARY_DECL_EXEC(O, I, P) \
using OutputType = O; using InputType = I; using ParamsType = P; using InstanceType = BinaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params)

    template <typename... OperationTypes>
    struct UnaryOperationSequence {
        UNARY_DECL_EXEC(typename FirstType_t<OperationTypes...>::InputType, typename LastType_t<OperationTypes...>::OutputType) {
            static_assert(std::is_same_v<typename FirstType_t<OperationTypes...>::InstanceType, UnaryType>);
            return UnaryOperationSequence<OperationTypes...>::next_exec<OperationTypes...>(input);
        }
    private:
        template <typename Operation>
        FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const typename Operation::InputType& input) {
            static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>);
            return Operation::exec(input);
        }
        template <typename Operation, typename... RemainingOperations>
        FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const typename Operation::InputType& input) {
            static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>);
            return UnaryOperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
        }
    };

    template <typename... Operations>
    struct OperationTupleOperation {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = typename FirstType_t<Operations...>::InstanceType;
    private:
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType> ||
                std::is_same_v<typename Operation::InstanceType, ReadType>) {
                return Operation::exec(i_data, tuple.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                return Operation::exec(i_data);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, WriteType>) {
                Operation::exec(i_data, tuple.params);
                return i_data;
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const typename Tuple_::Operation::InputType& i_data,
                                                     const Tuple_& tuple) {
            const auto result = exec_operate(i_data, tuple);
            if constexpr (Tuple_::size > 1) {
                return tuple_operate(result, tuple.next);
            } else {
                return result;
            }
        }
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& tuple) {
            return tuple_operate(input, tuple);
        }
    };

    template <typename Operation, typename I, typename O = I>
    struct UnaryV {
        UNARY_DECL_EXEC(I, O) {
            static_assert(cn<I> == cn<O>, "Unary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA = (validCUDAVec<I> && validCUDAVec<O>) || !(validCUDAVec<I> || validCUDAVec<O>);
            static_assert(allCUDAOrNotCUDA, "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<I> == 1) {
                if constexpr (validCUDAVec<I>) {
                    return { Operation::exec(input.x) };
                } else {
                    return Operation::exec(input);
                }
            } else if constexpr (cn<I> == 2) {
                return { Operation::exec(input.x),
                         Operation::exec(input.y) };

            } else if constexpr (cn<I> == 3) {
                return { Operation::exec(input.x),
                         Operation::exec(input.y),
                         Operation::exec(input.z) };
            } else {
                return { Operation::exec(input.x),
                         Operation::exec(input.y),
                         Operation::exec(input.z),
                         Operation::exec(input.w) };
            }
        }
    };

    template <typename Operation, typename I, typename P = I, typename O = I>
    struct BinaryV {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(cn<I> == cn<O>, "Binary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA = (validCUDAVec<I> && validCUDAVec<O>) || !(validCUDAVec<I> || validCUDAVec<O>);
            static_assert(allCUDAOrNotCUDA, "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<I> == 1) {
                if constexpr (validCUDAVec<I> && validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x) };
                } else if constexpr (validCUDAVec<I>) {
                    return { Operation::exec(input.x, params) };
                } else {
                    return Operation::exec(input, params);
                }
            } else if constexpr (cn<I> == 2) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params) };
                }

            } else if constexpr (cn<I> == 3) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y),
                             Operation::exec(input.z, params.z) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params),
                             Operation::exec(input.z, params) };
                }

            } else {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y),
                             Operation::exec(input.z, params.z),
                             Operation::exec(input.w, params.w) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params),
                             Operation::exec(input.z, params),
                             Operation::exec(input.w, params) };
                }
            }
        }
    };

    template <typename Operation, int ITERATIONS>
    struct StaticLoop {
        using InputType = typename Operation::InputType;
        using OutputType = typename Operation::OutputType;
        using ParamsType = typename Operation::ParamsType;
        using InstanceType = BinaryType;

    private:
        template <int ITERATION>
        FK_DEVICE_FUSE OutputType helper_exec(const InputType& input, const ParamsType& params) {
            if constexpr (ITERATION + 1 < ITERATIONS) {
                return helper_exec<ITERATION + 1>(Operation::exec(input, params), params);
            } else {
                return input;
            }
        }

    public:
        FK_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return helper_exec<0>(Operation::exec(input, params), params);
        }
    };
}//namespace fk
