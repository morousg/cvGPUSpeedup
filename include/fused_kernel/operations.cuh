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
#include "vlimits.cuh"

#include <climits>

namespace fk {

struct ReadType {};
struct WriteType {};
struct UnaryType {};
struct BinaryType {};
struct MidWriteType {};

#define UNARY_DECL_EXEC(I, O) \
using InputType = I; using OutputType = O; using InstanceType = UnaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input)

template <typename I, typename O> 
struct SaturateCast {
    UNARY_DECL_EXEC(I, O) {
        return saturate_cast<OutputType>(input);
    }
};

template <typename I, typename O>
struct Discard {
    UNARY_DECL_EXEC(I, O) {
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
};

template <typename T, int... idxs>
struct VectorReorder {
    UNARY_DECL_EXEC(T, T) {
        static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(cn<InputType> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        return VReorder<idxs...>::exec(input);
    }
};

template <typename... OperationTypes>
struct OperationSequence {
    UNARY_DECL_EXEC(typename FirstType_t<OperationTypes...>::InputType, typename LastType_t<OperationTypes...>::OutputType) {
        return OperationSequence<OperationTypes...>::next_exec<OperationTypes...>(input);
    }
private:
    template <typename Operation>
    FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const typename Operation::InputType& input) {
        return Operation::exec(input);
    }
    template <typename Operation, typename... RemainingOperations>
    FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const typename Operation::InputType& input) {
        return OperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
    }
};

enum GrayFormula { CCIR_601 };

template <typename I, typename O = VBase<I>, GrayFormula GF = CCIR_601>
struct RGB2Gray {};

template <typename I, typename O>
struct RGB2Gray<I, O, CCIR_601> {
public:
    UNARY_DECL_EXEC(I, O) {
        // 0.299*R + 0.587*G + 0.114*B
        if constexpr (std::is_unsigned_v<OutputType>) {
            return __float2uint_rn(compute_luminance(input));
        } else if constexpr (std::is_signed_v<OutputType>) {
            return __float2int_rn(compute_luminance(input));
        } else if constexpr (std::is_floating_point_v<OutputType>) {
            return compute_luminance(input);
        }
    }
private:
    FK_HOST_DEVICE_FUSE float compute_luminance(const InputType& input) {
        return (input.x * 0.299f) + (input.y * 0.587f) + (input.z * 0.114f);
    }
};

#define BINARY_DECL_EXEC(O, I, P) \
using OutputType = O; using InputType = I; using ParamsType = P; using InstanceType = BinaryType; \
static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params)


template <typename I, typename P = I, typename O = I>
struct Sum_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum_ can't work with cuda vector types.");
        return input + params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Sub_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sub_ can't work with cuda vector types.");
        return input - params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Mul_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Mul_ can't work with cuda vector types.");
        return input * params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Div_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Div_ can't work with cuda vector types.");
        return input / params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Max_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Max_ can't work with cuda vector types.");
        return input >= params ? input : params;
    }
};

template <typename I, typename P = I, typename O = I>
struct Min_ {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Min_ can't work with cuda vector types.");
        return input <= params ? input : params;
    }
};

enum ShiftDirection { Left, Right };

template <typename T, ShiftDirection SD>
struct Shift_ {
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
struct BinaryV {
    BINARY_DECL_EXEC(O, I, P) {
        static_assert(cn<I> == cn<O> && cn<I> == cn<P>, "Binary struct requires same number of channels for all types.");
        constexpr bool allAgregOrNotAgregate = std::is_aggregate_v<I> == std::is_aggregate_v<O> && std::is_aggregate_v<I> == std::is_aggregate_v<P>;
        static_assert(allAgregOrNotAgregate, "Binary struct requires all types to be agregate or all not agregate.");

        if constexpr (cn<I> == 1) {
            if constexpr (std::is_aggregate_v<I>) {
                return { Operation::exec(input.x, params.x) };
            } else {
                return Operation::exec(input, params);
            }
        } else if constexpr (cn<I> == 2) {
            return { Operation::exec(input.x, params.x),
                     Operation::exec(input.y, params.y) };
        } else if constexpr (cn<I> == 3) {
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
using Sum = BinaryV<Sum_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using Sub = BinaryV<Sub_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using Mul = BinaryV<Mul_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using Div = BinaryV<Div_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using Max = BinaryV<Max_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename I, typename P = I, typename O = I>
using Min = BinaryV<Min_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
template <typename T, ShiftDirection SD>
using Shift = BinaryV<Shift_<VBase<T>, SD>, T, VectorType_t<uint, cn<T>>>;
template <typename T>
using ShiftLeft = Shift<T, ShiftDirection::Left>;
template <typename T>
using ShiftRight = Shift<T, ShiftDirection::Right>;

#undef UNARY_DECL_EXEC

template <typename I, typename O>
struct AddLast {
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
struct SaturateFloat {
    using InputType = T;
    using OutputType = T;
    using Base = typename VectorTraits<T>::base;
    using InstanceType = UnaryType;

private:
    static constexpr __device__ __forceinline__ float saturate_channel(const float& input) {
        return Max<Base>::exec(0.f, Min<Base>::exec(input, 1.f));
    }

public:
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (std::is_same_v<InputType, float>) {
            return SaturateFloat<T>::saturate_channel(input);
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
    NONE = 17,
    INTER_LINEAR_CHANGE_NEIGHBOORS = -1
};

template <typename I, InterpolationType INTER_T>
struct Interpolate;

template <typename I>
struct Interpolate<I, InterpolationType::INTER_LINEAR> {
    using OutputType = VectorType_t<float, cn<I>>;
    using InputType = float2;
    using ParamsType = RawPtr<_2D, I>;
    using InstanceType = BinaryType;
    static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
        const float src_x = input.x;
        const float src_y = input.y;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = Min<int>::exec(x2, params.dims.width - 1);
        const int y2_read = Min<int>::exec(y2, params.dims.height - 1);

        
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

template <typename I>
struct Interpolate<I, InterpolationType::INTER_LINEAR_CHANGE_NEIGHBOORS> {
    using OutputType = VectorType_t<float, cn<I>>;
    using InputType = float2;
    using ParamsType = RawPtr<_2D, I>;
    using InstanceType = BinaryType;
    static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
        const float src_x = input.x;
        const float src_y = input.y;

        const int x1Tmp = __float2int_rd(src_x);
        const int y1Tmp = __float2int_rd(src_y);

        const bool x1Changed = src_x - x1Tmp < 0.5f;
        const bool y1Changed = src_y - y1Tmp < 0.5f;

        const int x1 = x1Changed ? Max<int>::exec(x1Tmp - 1, 0) : x1Tmp;
        const int y1 = y1Changed ? Max<int>::exec(y1Tmp - 1, 0) : y1Tmp;
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = Min<int>::exec(x2, params.dims.width - 1);
        const int y2_read = Min<int>::exec(y2, params.dims.height - 1);

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
enum ColorPrimitives { bt601, bt709 };

enum ColorDepth { p8bit, p10bit, p12bit };
template <ColorDepth CD>
struct CD_t { ColorDepth value{ CD }; };
using ColorDepthTypes = TypeList<CD_t<p8bit>, CD_t<p10bit>, CD_t<p12bit>>;
using ColorDepthChannelTypes = TypeList<uchar, ushort, ushort>;
using ColorDepthPixelTypes = TypeList<uchar3, ushort3, ushort3>;
template <ColorDepth CD>
using ColorDepthBase = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthChannelTypes>;

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

template <ColorDepth CD>
struct MaxDepthValue {};
template <> struct MaxDepthValue<p8bit> { enum { value = 255u }; };
template <> struct MaxDepthValue<p10bit> { enum { value = 1023u }; };
template <> struct MaxDepthValue<p12bit> { enum { value = 4095u }; };

struct MulCoefficients {
    float R1; float R2;
    float G1; float G2; float G3;
    float B1; float B2;
};

template <ColorRange CR, ColorPrimitives CP>
constexpr MulCoefficients mulCoefficients{};
template <> constexpr MulCoefficients mulCoefficients<Full,    bt601>{ 1.164f,  1.596f, 1.164f,  0.813f,  0.391f, 1.164f,  2.018f };
template <> constexpr MulCoefficients mulCoefficients<Limited, bt601>{ 1.164f,  1.596f, 1.164f,  0.813f,  0.391f, 1.164f,  2.018f };
template <> constexpr MulCoefficients mulCoefficients<Full,    bt709>{ 1.5748f, 0.f,    0.1873f, 0.4681f, 0.f,    1.8556f, 0.f    };
template <> constexpr MulCoefficients mulCoefficients<Limited, bt709>{ 1.4746f, 0.f,    0.1646f, 0.5713f, 0.f,    1.8814f, 0.f    };

template <typename T, ColorDepth CD>
struct SaturateDepth {
    using InputType = T;
    using OutputType = T;
    using InstanceType = UnaryType;
    using Base = typename VectorTraits<T>::base;

private:
    static constexpr __device__ __forceinline__ float saturate_channel(const float& input) {
        if constexpr (CD == p8bit) {
            return Max<Base>::exec(0.f, Min<Base>::exec(input, 255.f));
        } else if constexpr (CD == p10bit) {
            // TODO: check if this is correct, because if values are shifted to the most significant
            // bits, then, we have to consider the maximum value for ushort/uint16_t
            return Max<Base>::exec(0.f, Min<Base>::exec(input, 1023.f));
        } else if constexpr (CD == p12bit) {
            // TODO: check if this is correct, because if values are shifted to the most significant
            // bits, then, we have to consider the maximum value for ushort/uint16_t
            return Max<Base>::exec(0.f, Min<Base>::exec(input, 4095.f));
        }
    }

public:
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        if constexpr (std::is_same_v<InputType, float>) {
            return SaturateDepth<T, CD>::saturate_channel(input);
        } else {
            static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnarySaturateFloat");
            static_assert(std::is_same_v<Base, float>, "This function only works with floats");
            using I = T; using O = T;
            UNARY_EXECUTE_PER_CHANNEL(saturate_channel)
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

template <PixelFormat PF, ColorRange CR, ColorPrimitives CP, bool ALPHA>
struct ConvertYUVToRGB {
    static constexpr ColorDepth CD = (ColorDepth)PixelFormatTraits<PF>::depth;
    using InputType = YUVInputPixelType<CD>;
    using OutputType = YUVOutputPixelType<PF, ALPHA>;
    using InstanceType = UnaryType;

private:
    static constexpr ColorSpace CS = (ColorSpace)PixelFormatTraits<PF>::space;

    // Y -> input.x
    // U -> input.y
    // V -> input.z
    static constexpr __device__ __forceinline__ YUVChannelType<CD> computeR(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.R1* (input.x - subCoefficients<CD>.luma) + mulCoefficients<CR, CP>.R2* (input.z - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float V = input.z;
            const float RedMult = mulCoefficients<CR, CP>.R1;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::SaturateDepth<float, CD>::exec(Y + (RedMult * (V - ChromaSub)));
        }
    }
    static constexpr __device__ __forceinline__ YUVChannelType<CD> computeG(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.G1* (input.x - subCoefficients<CD>.luma) - mulCoefficients<CR, CP>.G2* (input.y - subCoefficients<CD>.chroma) - mulCoefficients<CR, CP>.G3* (input.z - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float U = input.y;
            const float V = input.z;
            const float GreenMul1 = mulCoefficients<CR, CP>.G1;
            const float GreenMul2 = mulCoefficients<CR, CP>.G2;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::SaturateDepth<float, CD>::exec(Y - (GreenMul1 * (U - ChromaSub)) - (GreenMul2 * (V - ChromaSub)));
        }
    }
    static constexpr __device__ __forceinline__ YUVChannelType<CD> computeB(const InputType& input) {
        if constexpr (CS == YUV420 && CP == bt601) {
            return mulCoefficients<CR, CP>.B1* (input.x - subCoefficients<CD>.luma) + mulCoefficients<CR, CP>.B2* (input.y - subCoefficients<CD>.chroma);
        } else if constexpr (CS == YUV420 && CP == bt709) {
            const float Y = input.x;
            const float U = input.y;
            const float BlueMul = mulCoefficients<CR, CP>.B1;
            const float ChromaSub = subCoefficients<CD>.chroma;

            return fk::SaturateDepth<float, CD>::exec(Y + (BlueMul * (U - ChromaSub)));
        }
    }
    static constexpr __device__ __forceinline__ InputType computeRGB(const InputType& pixel) {
        return { computeR(pixel), computeG(pixel), computeB(pixel) };
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
        using InputFactorType = typename ShiftRight<InputType>::ParamsType;
        const auto iShiftFactor = make_set<InputFactorType>(shiftFactor<CD>);
        // Pixel data shifted to the right to it's color depth numerical range
        const InputType shiftedPixel = ShiftRight<InputType>::exec(input, iShiftFactor);

        // Using color depth numerical ranged to comput the RGB pixel
        const OutputType computedPixel = computePixel(shiftedPixel);

        using OutputFactorType = typename ShiftLeft<OutputType>::ParamsType;
        const auto oShiftFactor = make_set<OutputFactorType>(shiftFactor<CD>);
        // Moving back the pixel channels to data type numerical range, either 8bit or 16bit
        return ShiftLeft<OutputType>::exec(computedPixel, oShiftFactor);
    }
};

template <typename I>
struct IsEven {
    using InputType = I;
    using OutputType = bool;
    using InstanceType = UnaryType;
    using AcceptedTypes = TypeList<uchar, ushort, uint>;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        static_assert( one_of_v<InputType, AcceptedTypes> ,"Input type not valid for UnaryIsEven");
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

} //namespace fk
