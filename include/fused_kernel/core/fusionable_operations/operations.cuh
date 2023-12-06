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

#include <climits>

namespace fk {

#define UNARY_DECL_EXEC(I, O) \
using InputType = I; using OutputType = O; using InstanceType = UnaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input)

#define BINARY_DECL_EXEC(O, I, P) \
using OutputType = O; using InputType = I; using ParamsType = P; using InstanceType = BinaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params)

    struct ReadType {};
    struct WriteType {};
    struct UnaryType {};
    struct BinaryType {};
    struct MidWriteType {};

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
    struct UnaryParams {};

    template <typename... Operations>
    struct BinaryParams {};

    using OpTypes = TypeList<UnaryType, BinaryType, WriteType, ReadType>;

    template <typename... Operations>
    using ParamTypes = TypeList<UnaryParams<Operations...>, BinaryParams<Operations...>, BinaryParams<Operations...>, BinaryParams<Operations...>>;

    template <typename T, typename... Operations>
    using NextType = EquivalentType_t<T, OpTypes, ParamTypes<Operations...>>;

    template <typename Operation>
    struct UnaryParams<Operation> {
        static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>, "Operation is not Unary");
    };

    template <typename Operation, typename... Operations>
    struct UnaryParams<Operation, Operations...> {
        static_assert(sizeof...(Operations) > 0, "Invalid specialization of Params");
        static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>, "Operation is not Unary");
        NextType<typename FirstType_t<Operations...>::InstanceType, Operations...> next;
    };

    template <typename Operation>
    struct BinaryParams<Operation> {
        static_assert(std::is_same_v<typename Operation::InstanceType, BinaryType>, "Operation is not Binary");
        typename Operation::ParamsType params;
    };

    template <typename Operation, typename... Operations>
    struct BinaryParams<Operation, Operations...> {
        static_assert(sizeof...(Operations) > 0, "Invalid specialization of Params");
        static_assert(std::is_same_v<typename Operation::InstanceType, BinaryType> ||
            std::is_same_v<typename Operation::InstanceType, WriteType> ||
            std::is_same_v<typename Operation::InstanceType, ReadType>, "Operation is not Binary, Write or Read");
        typename Operation::ParamsType params;
        NextType<typename FirstType_t<Operations...>::InstanceType, Operations...> next;
    };

    template <typename... Operations>
    struct ComposedOperationSequence {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = NextType<typename FirstType_t<Operations...>::InstanceType, Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = BinaryType;
    private:
        template <typename Operation, typename ComposedParamsType>
        FK_HOST_DEVICE_FUSE auto exec_operate(const typename Operation::InputType& i_data, const ComposedParamsType& head) {
            if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType> ||
                std::is_same_v<typename Operation::InstanceType, ReadType>) {
                return Operation::exec(i_data, head.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                return Operation::exec(i_data);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, WriteType>) {
                Operation::exec(i_data, head.params);
                return i_data;
            }
        }
        template <typename ComposedParamsType, typename Operation, typename... OperationTypes>
        FK_HOST_DEVICE_FUSE OutputType composed_operate(const typename Operation::InputType& i_data,
            const ComposedParamsType& head) {
            if constexpr (sizeof...(OperationTypes) > 0) {
                using NextComposedParamsType = decltype(head.next);
                const auto result = exec_operate<Operation, ComposedParamsType>(i_data, head);
                return composed_operate<NextComposedParamsType, OperationTypes...>(result, head.next);
            } else {
                return exec_operate<Operation>(i_data, head);
            }
        }
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& head) {
            return ComposedOperationSequence<Operations...>::composed_operate<ParamsType, Operations...>(input, head);
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

    template <typename I>
    struct IsEven {
        using InputType = I;
        using OutputType = bool;
        using InstanceType = UnaryType;
        using AcceptedTypes = TypeList<uchar, ushort, uint>;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(one_of_v<InputType, AcceptedTypes>, "Input type not valid for UnaryIsEven");
            return (input & 1u) == 0;
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
