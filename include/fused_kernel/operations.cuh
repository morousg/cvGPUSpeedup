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

#include <climits>

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
    FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const typename Operation::InputType& input) {
        return Operation::exec(input);
    }
    template <typename Operation, typename... RemainingOperations>
    FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const typename Operation::InputType& input) {
        return UnaryOperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
    }
};

#undef UNARY_DECL_EXEC

#define BINARY_DECL_EXEC(O, I, P) \
using OutputType = O; using InputType = I; using ParamsType = P; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params)


template <typename I, typename P = I, typename O = I>
struct Sum {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input + params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Sub {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input - params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Mul {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input * params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Div {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input / params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Max {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input >= params ? input : params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Min {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum can't work with cuda vector types.");
        return input <= params ? input : params;
    }
};

enum ShiftDirection { Left, Right };

template <typename T, ShiftDirection SD>
struct Shift {
    BINARY_DECL_EXEC(T, T, uint) {
        static_assert(!validCUDAVec<T>, "Shift can't work with cuda vector types.");
        static_assert(std::is_unsigned_v<T>, "Shift only works with unsigned integers.");
        if constexpr (SD == Left) {
            return input << params;
        } else if constexpr (SD == Right) {
            return input >> params;
        }
    }
};

template <typename Operation, typename I, typename P = I, typename O = I>
struct Binary {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(cn<I> == cn<O> && cn<I> == cn<P>, "Binary struct requires same number of channels for all types.");
        constexpr bool allAgregOrNotAgregate = std::is_aggregate_v<I> == std::is_aggregate_v<O> && std::is_aggregate_v<I> == std::is_aggregate_v<P>;
        static_assert(allAgregOrNotAgregate, "Binary struct requires all types to be agregate or all not agregate.");

        constexpr int CN = cn<I>;
        if constexpr (CN == 1) {
            if constexpr (std::is_aggregate_v<I>) {
                return { Operation::exec(input.x, params.x) };
            } else {
                return Operation::exec(input, params);
            }
        } else if constexpr (CN == 2) {
            return { Operation::exec(input.x, params.x),
                     Operation::exec(input.y, params.y) };
        } else if constexpr (CN == 3) {
            return { Operation::exec(input.x, params.x),
                     Operation::exec(input.y, params.y),
                     Operation::exec(input.z, params.z) };
        } else {
            return { Operation::exec(input.x, params.x),
                     Operation::exec(input.y, params.y),
                     Operation::exec(input.z, params.z),
                     Operation::exec(input.w, params.w) };
        }
    }
};

template <typename I, typename P = I, typename O = I>
using BinarySum = Binary<Sum<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using BinarySub = Binary<Sub<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using BinaryMul = Binary<Mul<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using BinaryDiv = Binary<Div<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using BinaryMax = Binary<Max<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using BinaryMin = Binary<Min<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename T, ShiftDirection SD>
using BinaryShift = Binary<Shift<VBase<T>, SD>, T>;

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

#undef BINARY_DECL_EXEC

template <typename T>
struct UnarySaturateFloat {
    using InputType = T;
    using OutputType = T;
    using Base = typename VectorTraits<T>::base;

    static constexpr __device__ __forceinline__ float saturate_channel(const float& input) {
        return BinaryMax<Base>::exec(0.f, BinaryMin<Base>::exec(input, 1.f));
    }

    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (std::is_same_v<InputType, float>) {
            return UnarySaturateFloat<T>::saturate_channel(input);
        } else {
            static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnarySaturateFloat");
            static_assert(std::is_same_v<Base, float>, "This function only works with floats");
            using I = T; using O = T;
            UNARY_EXECUTE_PER_CHANNEL(saturate_channel)
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
    using OutputType = VectorType_t<float, cn<I>>;
    using InputType = float2;
    using ParamsType = RawPtr<_2D, I>;
    static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
        const float src_x = input.x;
        const float src_y = input.y;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = BinaryMin<int>::exec(x2, params.dims.width - 1);
        const int y2_read = BinaryMin<int>::exec(y2, params.dims.height - 1);

        
        const I src_reg0x0 = *PtrAccessor<_2D>::cr_point(Point(x1, y1), params);
        const I src_reg1x0 = *PtrAccessor<_2D>::cr_point(Point(x2_read, y1), params);
        const I src_reg0x1 = *PtrAccessor<_2D>::cr_point(Point(x1, y2_read), params);
        const I src_reg1x1 = *PtrAccessor<_2D>::cr_point(Point(x2_read, y2_read), params);

        const OutputType out = (src_reg0x0 * ((x2 - src_x) * (y2 - src_y))) +
                               (src_reg1x0 * ((src_x - x1) * (y2 - src_y))) +
                               (src_reg0x1 * ((x2 - src_x) * (src_y - y1))) +
                               (src_reg1x1 * ((src_x - x1) * (src_y - y1)));
        return out;
    }
};

enum ColorSpace { YUV420, YUV422, YUV444 };
template <ColorSpace CS>
struct CS_t { ColorSpace value{ CS }; };

enum ColorRange { Limited, Full };
enum ColorPrimitves { bt601, bt709 };

enum ColorDepth { p8bit, p10bit, p12bit };
template <ColorDepth CD>
struct CD_t { ColorDepth value{ CD }; };
using ColorDepthTypes = TypeList<CD_t<p8bit>, CD_t<p10bit>, CD_t<p12bit>>;
using ColorDepthChannelTypes = TypeList<uchar, ushort, ushort>;
using ColorDepthPixelTypes = TypeList<uchar3, ushort3, ushort3>;

enum PixelFormat { NV12, NV21, YV12, P010, P016, P216, P210, Y216, Y210, Y416 };
template <PixelFormat PF>
struct PixelFormatTraits;
template <> struct PixelFormatTraits<NV12> { enum { space = YUV420, depth = p8bit, cn = 3 }; };
template <> struct PixelFormatTraits<NV21> { enum { space = YUV420, depth = p8bit, cn = 3 }; };
template <> struct PixelFormatTraits<YV12> { enum { space = YUV420, depth = p8bit, cn = 3 }; };
template <> struct PixelFormatTraits<P010> { enum { space = YUV420, depth = p10bit, cn = 3 }; };
template <> struct PixelFormatTraits<P210> { enum { space = YUV422, depth = p10bit, cn = 3 }; };
template <> struct PixelFormatTraits<Y210> { enum { space = YUV422, depth = p10bit, cn = 3 }; };
template <> struct PixelFormatTraits<Y416> { enum { space = YUV444, depth = p12bit, cn = 4 }; };

template <ColorDepth CD>
using YUVChannelType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthChannelTypes>;
template <ColorDepth CD>
using YUVInputPixelType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelTypes>;
template <PixelFormat PF, bool ALPHA>
using YUVOutputPixelType = VectorType_t<YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>, ALPHA ? 4 : PixelFormatTraits<PF>::cn>;

struct SubCoefficients {
    float luma;
    float chroma;
};

template <ColorDepth CD>
constexpr SubCoefficients subCoefficients{};
template <> constexpr SubCoefficients subCoefficients<p8bit>{ 16.f, 128.f };
template <> constexpr SubCoefficients subCoefficients<p10bit>{ 64.f, 512.f };
template <> constexpr SubCoefficients subCoefficients<p12bit>{ 64.f, 2048.f };

struct MulCoefficients {
    float R1; float R2;
    float G1; float G2; float G3;
    float B1; float B2;
};

template <ColorRange CR, ColorPrimitves CP>
constexpr MulCoefficients mulCoefficients{};
template <> constexpr MulCoefficients mulCoefficients<Full,    bt601>{ 1.164f,  1.596f, 1.164f,  0.813f,  0.391f, 1.164f,  2.018f };
template <> constexpr MulCoefficients mulCoefficients<Limited, bt601>{ 1.164f,  1.596f, 1.164f,  0.813f,  0.391f, 1.164f,  2.018f };
template <> constexpr MulCoefficients mulCoefficients<Full,    bt709>{ 1.5748f, 0.f,    0.1873f, 0.4681f, 0.f,    1.8556f, 0.f    };
template <> constexpr MulCoefficients mulCoefficients<Limited, bt709>{ 1.4746f, 0.f,    0.1646f, 0.5713f, 0.f,    1.8814f, 0.f    };

template <typename T, ColorDepth CD>
struct UnarySaturate {
    using InputType = T;
    using OutputType = T;
    using Base = typename VectorTraits<T>::base;

    static constexpr __device__ __forceinline__ float saturate_channel(const float& input) {
        if constexpr (CD == p8bit) {
            return BinaryMax<Base>::exec(0.f, BinaryMin<Base>::exec(input, 255.f));
        } else if constexpr (CD == p10bit) {
            // TODO: check if this is correct, because if values are shifted to the most significant
            // bits, then, we have to consider the maximum value for ushort/uint16_t
            return BinaryMax<Base>::exec(0.f, BinaryMin<Base>::exec(input, 1023.f));
        } else if constexpr (CD == p12bit) {
            // TODO: check if this is correct, because if values are shifted to the most significant
            // bits, then, we have to consider the maximum value for ushort/uint16_t
            return BinaryMax<Base>::exec(0.f, BinaryMin<Base>::exec(input, 4095.f));
        }
    }

    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (std::is_same_v<InputType, float>) {
            return UnarySaturate<T, CD>::saturate_channel(input);
        }
        else {
            static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnarySaturateFloat");
            static_assert(std::is_same_v<Base, float>, "This function only works with floats");
            using I = T; using O = T;
            UNARY_EXECUTE_PER_CHANNEL(saturate_channel)
        }
    }
};

// Y -> input.x
// U -> input.y
// V -> input.z
template <ColorSpace CS, ColorDepth CD, ColorRange CR, ColorPrimitves CP>
struct ComputeR {
    using InputType = YUVInputPixelType<CD>;
    using OutputType = YUVChannelType<CD>;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.R1 * (input.x - subCoefficients<CD>.luma) + mulCoefficients<CR, CP>.R2 * (input.z - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float V = input.z;
            const float RedMult = mulCoefficients<CR, CP>.R1;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::UnarySaturate<float, CD>::exec(Y + (RedMult * (V - ChromaSub)));
        }
    }
};

template <ColorSpace CS, ColorDepth CD, ColorRange CR, ColorPrimitves CP>
struct ComputeG {
    using InputType = YUVInputPixelType<CD>;
    using OutputType = YUVChannelType<CD>;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.G1 * (input.x - subCoefficients<CD>.luma) - mulCoefficients<CR, CP>.G2 * (input.y - subCoefficients<CD>.chroma) - mulCoefficients<CR, CP>.G3 * (input.z - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float U = input.y;
            const float V = input.z;
            const float GreenMul1 = mulCoefficients<CR, CP>.G1;
            const float GreenMul2 = mulCoefficients<CR, CP>.G2;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::UnarySaturate<float, CD>::exec(Y - (GreenMul1 * (U - ChromaSub)) - (GreenMul2 * (V - ChromaSub)));
        }
    }
};

template <ColorSpace CS, ColorDepth CD, ColorRange CR, ColorPrimitves CP>
struct ComputeB {
    using InputType = YUVInputPixelType<CD>;
    using OutputType = YUVChannelType<CD>;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.B1 * (input.x - subCoefficients<CD>.luma) + mulCoefficients<CR, CP>.B2 * (input.y - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float U = input.y;
            const float BlueMul = mulCoefficients<CR, CP>.B1;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::UnarySaturate<float, CD>::exec(Y + (BlueMul * (U - ChromaSub)));
        }
    }
};

template <ColorDepth CD> struct AlphaValue { YUVChannelType<CD> value; };
template <ColorDepth CD> constexpr AlphaValue<CD> alphaValue{};
template <> constexpr AlphaValue<p8bit> alphaValue<p8bit>{ 255u };
template <> constexpr AlphaValue<p10bit> alphaValue<p10bit>{ 1023u };
template <> constexpr AlphaValue<p12bit> alphaValue<p12bit>{ 4095u };

template <ColorDepth CD> constexpr YUVChannelType<CD> shiftFactor{};
template <> constexpr YUVChannelType<p8bit> shiftFactor<p8bit>{ 0u };
template <> constexpr YUVChannelType<p10bit> shiftFactor<p10bit>{ 6u };
template <> constexpr YUVChannelType<p12bit> shiftFactor<p12bit>{ 4u };

template <PixelFormat PF, ColorRange CR, ColorPrimitves CP, bool ALPHA>
struct UnaryConvertYUVToRGB {
    using InputType = YUVInputPixelType<(ColorDepth)PixelFormatTraits<PF>::depth>;
    using OutputType = YUVOutputPixelType<PF, ALPHA>;

private:
    static constexpr ColorSpace CS = (ColorSpace)PixelFormatTraits<PF>::space;
    static constexpr ColorDepth CD = (ColorDepth)PixelFormatTraits<PF>::depth;
    static constexpr __device__ __forceinline__ InputType computeRGB(const InputType& pixel) {
        return { ComputeR<CS, CD, CR, CP>::exec(pixel),
                 ComputeG<CS, CD, CR, CP>::exec(pixel),
                 ComputeB<CS, CD, CR, CP>::exec(pixel) };
    }

    static constexpr __device__ __forceinline__ OutputType computePixel(const InputType& pixel) {
        const InputType pixelRGB = computeRGB(pixel);
        if constexpr (ALPHA) {
            return { pixelRGB.x, pixelRGB.y, pixelRGB.z, alphaValue<CD>.value };
        } else {
            return pixelRGB;
        }
    }

public:
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        const InputType shiftedPixel = BinaryShift<InputType, ShiftDirection::Right>::exec(input, make_set<InputType>(shiftFactor<CD>));
        const OutputType computedPixel = computePixel(shiftedPixel);
        return BinaryShift<OutputType, ShiftDirection::Left>::exec(computedPixel, make_set<OutputType>(shiftFactor<CD>));
    }
};

template <typename I>
struct UnaryIsEven {
    using InputType = I;
    using OutputType = bool;
    using AcceptedTypes = TypeList<uchar, ushort, uint>;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        static_assert( one_of_v<InputType, AcceptedTypes> ,"Input type not valid for UnaryIsEven");
        return (input & 1u) == 0;
    }
};
} //namespace fk
