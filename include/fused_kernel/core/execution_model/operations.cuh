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
#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/utils/tuple.cuh>
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

    template <typename Operation, typename Enabler=void>
    constexpr bool getThreadFusion{};

    template <typename Operation>
    constexpr bool getThreadFusion<Operation, std::enable_if_t<!isReadType<Operation>>>{ false };

    template <typename Operation>
    constexpr bool getThreadFusion<Operation, std::enable_if_t<isReadType<Operation>>>{ Operation::THREAD_FUSION };

    template <typename Operation, typename Enabler=void>
    struct GetThreadFusionInfo {};

    template <typename Operation>
    struct GetThreadFusionInfo<Operation, std::enable_if_t<!isReadType<Operation>>> {
        using type = void;
    };

    template <typename Operation>
    struct GetThreadFusionInfo<Operation, std::enable_if_t<isReadType<Operation>>> {
        using type = typename Operation::ThreadFusion;
    };

    struct OperationTupleOperationImpl {
        private:
            template <typename Tuple_>
            FK_HOST_DEVICE_FUSE auto exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
                using Operation = typename Tuple_::Operation;
                if constexpr (std::is_same_v<typename Operation::InstanceType, TernaryType>) {
                    return Operation::exec(i_data, tuple.params, tuple.back_function);
                } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                    return Operation::exec(i_data, tuple.params);
                } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                    return Operation::exec(i_data);
                // Assuming the behavior of a MidWriteType DeviceFunction
                } else if constexpr (std::is_same_v<typename Operation::InstanceType, WriteType>) {
                    Operation::exec(i_data, tuple.params);
                    return i_data;
                }
            }
            template <typename Tuple_>
            FK_HOST_DEVICE_FUSE auto exec_operate(const Point& thread, const Tuple_& tuple) {
                using Operation = typename Tuple_::Operation;
                if constexpr (std::is_same_v<typename Operation::InstanceType, ReadType>) {
                    return Operation::exec(thread, tuple.params);
                } else if constexpr (std::is_same_v<typename Operation::InstanceType, ReadBackType>) {
                    return Operation::exec(thread, tuple.params, tuple.back_function);
                }
            }
        public:
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
            template <typename Tuple_>
            FK_HOST_DEVICE_FUSE auto tuple_operate(const Point& thread, const Tuple_& tuple) {
                const auto result = exec_operate(thread, tuple);
                if constexpr (Tuple_::size > 1) {
                    return tuple_operate(result, tuple.next);
                } else {
                    return result;
                }
            }
    };
    using OTOImpl = OperationTupleOperationImpl;

    template <typename Enabler, typename... Operations>
    struct OperationTupleOperation_ {};

    template <typename... Operations>
    struct OperationTupleOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = BinaryType;

    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& tuple) {
            return OTOImpl::tuple_operate(input, tuple);
        }
    };

    template <typename... Operations>
    struct OperationTupleOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        static constexpr bool THREAD_FUSION{ FirstType_t<Operations...>::THREAD_FUSION };
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return OTOImpl::tuple_operate(thread, tuple);
        }
    };

    template <typename... Operations>
    using OperationTupleOperation = OperationTupleOperation_<void, Operations...>;

    template <typename I, typename O, typename Operation>
    struct UnaryV {
        UNARY_DECL_EXEC(I, O) {
            static_assert(cn<InputType> == cn<OutputType>, "Unary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA = (validCUDAVec<InputType> && validCUDAVec<OutputType>) ||
                                             !(validCUDAVec<InputType> || validCUDAVec<OutputType>);
            static_assert(allCUDAOrNotCUDA, "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<InputType> == 1) {
                if constexpr (validCUDAVec<InputType>) {
                    return { Operation::exec(input.x) };
                } else {
                    return Operation::exec(input);
                }
            } else if constexpr (cn<InputType> == 2) {
                return { Operation::exec(input.x),
                         Operation::exec(input.y) };

            } else if constexpr (cn<InputType> == 3) {
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
        template <int ITERATION>
        FK_DEVICE_FUSE OutputType helper_exec(const InputType& input) {
            if constexpr (ITERATION + 1 < ITERATIONS) {
                return helper_exec<ITERATION + 1>(Operation::exec(input));
            } else {
                return input;
            }
        }

    public:
        FK_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return helper_exec<0>(Operation::exec(input, params), params);
        }
        FK_DEVICE_FUSE OutputType exec(const InputType& input) {
            return helper_exec<0>(Operation::exec(input));
        }
    };

    template <typename I, typename O>
    struct CastBase {
        UNARY_DECL_EXEC(I, O) {
            return static_cast<O>(input);
        }
    };

    template <typename I, typename O>
    struct Cast {
        UNARY_DECL_EXEC(I, O) {
            return UnaryV<I,O,CastBase<VBase<I>,VBase<O>>>::exec(input);
        }
    };

} //namespace fk
