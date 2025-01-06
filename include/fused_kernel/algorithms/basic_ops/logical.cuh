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

#ifndef FK_LOGICAL
#define FK_LOGICAL

#include <fused_kernel/core/execution_model/vector_operations.cuh>
#include <fused_kernel/core/data/tuple.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    enum ShiftDirection { Left, Right };

    template <typename T, ShiftDirection SD>
    struct ShiftBase {
        using OutputType = T;
        using InputType = T;
        using ParamsType = uint;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<ShiftBase<T, SD>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            static_assert(!validCUDAVec<T>, "Shift can't work with cuda vector types.");
            static_assert(std::is_unsigned_v<T>, "Shift only works with unsigned integers.");
            if constexpr (SD == Left) {
                return input << opData.params;
            } else if constexpr (SD == Right) {
                return input >> opData.params;
            }
        }
    };

    template <typename T, ShiftDirection SD>
    struct Shift {
        using OutputType = T;
        using InputType = T;
        using ParamsType = uint;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<Shift<T, SD>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return BinaryV<ShiftBase<VBase<T>, SD>, T, uint>::exec(input, { opData.params });
        }
        using InstantiableType = Binary<BinaryV<ShiftBase<VBase<T>, SD>, T, uint>>;
        DEFAULT_BUILD
    };
    template <typename T>
    using ShiftLeft = Shift<T, ShiftDirection::Left>;
    template <typename T>
    using ShiftRight = Shift<T, ShiftDirection::Right>;

    template <typename I>
    struct IsEven {
        using InputType = I;
        using OutputType = bool;
        using InstanceType = UnaryType;
        using AcceptedTypes = TypeList<uchar, ushort, uint>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(one_of_v<InputType, AcceptedTypes>, "Input type not valid for UnaryIsEven");
            return (input & 1u) == 0;
        }
        using InstantiableType = Unary<IsEven<I>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MaxBase {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<MaxBase<I, P, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Max_ can't work with cuda vector types.");
            return input >= opData.params ? input : opData.params;
        }
    };

    template <typename I, typename P, typename O>
    struct MaxBase<I, P, O, UnaryType> {
        using OutputType = O;
        using InputType = Tuple<I, P>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Max_ can't work with cuda vector types.");
            return get<0>(input) >= get<1>(input) ? get<0>(input) : get<1>(input);
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Max {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<Max<I, P, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return BinaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, opData);
        }
        using InstantiableType = Binary<BinaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>>;
        DEFAULT_BUILD
    };

    template <typename I, typename P, typename O>
    struct Max<I, P, O, UnaryType> {
        using OutputType = O;
        using InputType = Tuple<I,P>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, InputType, O>::exec(input);
        }
        using InstantiableType = Unary<UnaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, InputType, O>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MinBase {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<MinBase<I, P, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Min_ can't work with cuda vector types.");
            return input <= opData.params ? input : opData.params;
        }
    };

    template <typename I, typename P, typename O>
    struct MinBase<I, P, O, UnaryType> {
        using OutputType = O;
        using InputType = Tuple<I,P>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Min_ can't work with cuda vector types.");
            return get<0>(input) <= get<1>(input) ? get<0>(input) : get<1>(input);
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Min {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<Min<I, P, O>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return BinaryV<MinBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, { opData.params });
        }
        using InstantiableType = Binary<BinaryV<MinBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>>;
        DEFAULT_BUILD
    };

    template <typename I, typename P, typename O>
    struct Min<I, P, O, UnaryType> {
        using OutputType = O;
        using InputType = Tuple<I,P>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MinBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, InputType, O>::exec(input);
        }
        using InstantiableType = Unary<UnaryV<MinBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, InputType, O>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename I1, typename I2=I1>
    struct Equal {
        using OutputType = bool;
        using InputType = Tuple<I1,I2>;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) == get<1>(input);
        }
        using InstantiableType = Unary<Equal<I1, I2>>;
        DEFAULT_UNARY_BUILD
    };
} //namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
