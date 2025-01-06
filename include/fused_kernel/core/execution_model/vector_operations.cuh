/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_VECTOR_OPERATIONS
#define FK_VECTOR_OPERATIONS

#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>

namespace fk {
    template <typename Operation, typename I, typename O, typename Enabler = void>
    struct UnaryV {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(cn<InputType> == cn<OutputType>,
                "Unary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA =
                (validCUDAVec<InputType> && validCUDAVec<OutputType>) ||
                !(validCUDAVec<InputType> || validCUDAVec<OutputType>);
            static_assert(allCUDAOrNotCUDA,
                "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

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

    template <typename Operation, typename I, typename O>
    struct UnaryV<Operation, I, O, std::enable_if_t<isTuple_v<I>, void>> {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const auto input1 = get<0>(input);
            const auto input2 = get<1>(input);
            using I1 = get_t<0, I>;
            using I2 = get_t<1, I>;
            static_assert(cn<I1> == cn<O>,
                "Binary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA =
                (validCUDAVec<I1> && validCUDAVec<O>) || !(validCUDAVec<I1> || validCUDAVec<O>);
            static_assert(allCUDAOrNotCUDA,
                "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<I1> == 1) {
                if constexpr (validCUDAVec<I1> && validCUDAVec<I2>) {
                    return { Operation::exec({ input1.x, input2.x }) };
                } else if constexpr (validCUDAVec<I1>) {
                    return { Operation::exec({ input1.x, input2 }) };
                } else {
                    return Operation::exec({ input1, input2 });
                }
            } else if constexpr (cn<I1> == 2) {
                if constexpr (validCUDAVec<I2>) {
                    return { Operation::exec({ input1.x, input2.x }),
                             Operation::exec({ input1.y, input2.y }) };
                } else {
                    return { Operation::exec({ input1.x, input2 }),
                             Operation::exec({ input1.y, input2 }) };
                }
            } else if constexpr (cn<I1> == 3) {
                if constexpr (validCUDAVec<I2>) {
                    return { Operation::exec({ input1.x, input2.x }),
                             Operation::exec({ input1.y, input2.y }),
                             Operation::exec({ input1.z, input2.z }) };
                } else {
                    return { Operation::exec({ input1.x, input2 }),
                             Operation::exec({ input1.y, input2 }),
                             Operation::exec({ input1.z, input2 }) };
                }
            } else {
                if constexpr (validCUDAVec<I2>) {
                    return { Operation::exec({ input1.x, input2.x }),
                             Operation::exec({ input1.y, input2.y }),
                             Operation::exec({ input1.z, input2.z }),
                             Operation::exec({ input1.w, input2.w }) };
                } else {
                    return { Operation::exec({ input1.x, input2 }),
                             Operation::exec({ input1.y, input2 }),
                             Operation::exec({ input1.z, input2 }),
                             Operation::exec({ input1.w, input2 }) };
                }
            }
        }
    };

    template <typename Operation, typename I, typename P = I, typename O = I>
    struct BinaryV {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<BinaryV<Operation, I, P, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            static_assert(cn<I> == cn<O>,
                "Binary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA =
                (validCUDAVec<I> && validCUDAVec<O>) || !(validCUDAVec<I> || validCUDAVec<O>);
            static_assert(allCUDAOrNotCUDA,
                "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<I> == 1) {
                if constexpr (validCUDAVec<I> && validCUDAVec<P>) {
                    return { Operation::exec(input.x, { opData.params.x }) };
                } else if constexpr (validCUDAVec<I>) {
                    return { Operation::exec(input.x, { opData.params }) };
                } else {
                    return Operation::exec(input, { opData.params });
                }
            } else if constexpr (cn<I> == 2) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, { opData.params.x }),
                             Operation::exec(input.y, { opData.params.y }) };
                } else {
                    return { Operation::exec(input.x, { opData.params }),
                             Operation::exec(input.y, { opData.params }) };
                }
            } else if constexpr (cn<I> == 3) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, { opData.params.x }),
                             Operation::exec(input.y, { opData.params.y }),
                             Operation::exec(input.z, { opData.params.z }) };
                } else {
                    return { Operation::exec(input.x, { opData.params }),
                             Operation::exec(input.y, { opData.params }),
                             Operation::exec(input.z, { opData.params }) };
                }
            } else {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, { opData.params.x }),
                             Operation::exec(input.y, { opData.params.y }),
                             Operation::exec(input.z, { opData.params.z }),
                             Operation::exec(input.w, { opData.params.w }) };
                } else {
                    return { Operation::exec(input.x, { opData.params }),
                             Operation::exec(input.y, { opData.params }),
                             Operation::exec(input.z, { opData.params }),
                             Operation::exec(input.w, { opData.params }) };
                }
            }
        }
    };
} // namespace fk

#endif
