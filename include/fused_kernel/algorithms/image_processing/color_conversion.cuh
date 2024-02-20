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

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/algorithms/basic_ops/logical.cuh>
#include <fused_kernel/algorithms/basic_ops/algebraic.cuh>
#include <fused_kernel/algorithms/image_processing/saturate.cuh>

namespace fk {
    template <typename I>
    struct AddAlpha {
        using InputType = I;
        using OutputType = VectorType_t<VBase<I>, (cn<I> +1)>;
        using InstanceType = UnaryType;
        FK_DEVICE_FUSE OutputType exec(const InputType& input) {
            if constexpr (std::is_same_v<VBase<InputType>, float>) {
                return AddLast<InputType, OutputType>::exec(input, 1.f);
            } else {
                constexpr VBase<I> alpha = maxValue<VBase<I>>;
                return AddLast<InputType, OutputType>::exec(input, alpha);
            }
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

    enum ColorSpace { YUV420, YUV422, YUV444 };
    template <ColorSpace CS>
    struct CS_t { ColorSpace value{ CS }; };

    enum ColorRange { Limited, Full };
    enum ColorPrimitives { bt601, bt709, bt2020 };

    enum ColorDepth { p8bit, p10bit, p12bit };
    template <ColorDepth CD>
    struct CD_t { ColorDepth value{ CD }; };
    using ColorDepthTypes = TypeList<CD_t<p8bit>, CD_t<p10bit>, CD_t<p12bit>>;
    using ColorDepthPixelBaseTypes = TypeList<uchar, ushort, ushort>;
    using ColorDepthPixelTypes = TypeList<uchar3, ushort3, ushort3>;
    template <ColorDepth CD>
    using ColorDepthPixelBaseType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelBaseTypes>;
    template <ColorDepth CD>
    using ColorDepthPixelType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelTypes>;


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

    template <PixelFormat PF, bool ALPHA>
    using YUVOutputPixelType = VectorType_t<ColorDepthPixelBaseType<(ColorDepth)PixelFormatTraits<PF>::depth>, ALPHA ? 4 : PixelFormatTraits<PF>::cn>;

    struct SubCoefficients {
        const float luma;
        const float chroma;
    };

    template <ColorDepth CD>
    constexpr SubCoefficients subCoefficients{};
    template <> constexpr SubCoefficients subCoefficients<p8bit>{ 16.f, 128.f };
    template <> constexpr SubCoefficients subCoefficients<p10bit>{ 64.f, 512.f };
    template <> constexpr SubCoefficients subCoefficients<p12bit>{ 64.f, 2048.f };

    template <ColorDepth CD>
    constexpr ColorDepthPixelBaseType<CD> maxDepthValue{};
    template <> constexpr ColorDepthPixelBaseType<p8bit>  maxDepthValue<p8bit> { 255u };
    template <> constexpr ColorDepthPixelBaseType<p10bit> maxDepthValue<p10bit> { 1023u };
    template <> constexpr ColorDepthPixelBaseType<p12bit> maxDepthValue<p12bit> { 4095u };

    template <typename T, ColorDepth CD>
    struct SaturateDepth {
        using InputType = T;
        using OutputType = T;
        using InstanceType = UnaryType;
        using Base = typename VectorTraits<T>::base;

        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            if constexpr (CD == p8bit) {
                return Saturate<float>::exec(input, {0.f, 255.f});
            } else if constexpr (CD == p10bit) {
                return Saturate<float>::exec(input, {0.f, 1023.f});
            } else if constexpr (CD == p12bit) {
                return Saturate<float>::exec(input, {0.f, 4095.f});
            }
        }
    };

    enum ColorConversionDir { YCbCr2RGB, RGB2YCbCr };

    template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CCD>
    constexpr M3x3Float ccMatrix{};
    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<Full, bt601, YCbCr2RGB>{
        { 1.164383562f,           0.f,       1.596026786f  },
        { 1.164383562f,  -0.39176229f,       -0.812967647f },
        { 1.164383562f,  2.017232143f,       0.f           }};

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<Full, bt709, YCbCr2RGB>{
        { 1.f,               0.f,            1.5748f },
        { 1.f,          -0.1873f,           -0.4681f },
        { 1.f,           1.8556f,                0.f }};

    // To be verified
    template <> constexpr M3x3Float ccMatrix<Limited, bt709, YCbCr2RGB>{
        { 1.f,               0.f,            1.402f },
        { 1.f,         -0.34414f,         -0.71414f },
        { 1.f,            1.772f,               0.f }};
        /*{  1.f,              0.f,            1.4746f },
        { 1.f,          -0.1646f,           -0.5713f },
        { 1.f,           1.8814f,            0.f     }};*/

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<Full, bt709, RGB2YCbCr>{
        {  0.2126f, 0.7152f, 0.0722f           },
        { -0.1146f,      -0.3854f,            0.5f },
        { 0.5f,         -0.4542f,           -0.0458f }};

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<Full, bt2020, YCbCr2RGB>{
        {  1.f, 0.f, 1.4746f           },
        { 1.f,          -0.16455312684366f, -0.57135312684366f },
        { 1.f,           1.8814f,            0.f }};

    // Computed from ccMatrix<Full, bt2020, YCbCr2RGB>
    template <> constexpr M3x3Float ccMatrix<Full, bt2020, RGB2YCbCr>{
        { -0.73792134831461f, 1.90449438202248f, -0.16657303370787f },
        { 0.39221927730127f, -1.01227510472121f,  0.62005582741994f },
        { 1.17857137414527f, -1.29153287808387f,  0.11296150393861f }};

    template <ColorDepth CD> constexpr ColorDepthPixelBaseType<CD> shiftFactor{};
    template <> constexpr ColorDepthPixelBaseType<p8bit>  shiftFactor<p8bit>{ 0u };
    template <> constexpr ColorDepthPixelBaseType<p10bit> shiftFactor<p10bit>{ 6u };
    template <> constexpr ColorDepthPixelBaseType<p12bit> shiftFactor<p12bit>{ 4u };

    template <ColorDepth CD>
    constexpr float floatShiftFactor{};
    template <> constexpr float floatShiftFactor<p8bit>{ 1.f };
    template <> constexpr float floatShiftFactor<p10bit>{ 64.f };
    template <> constexpr float floatShiftFactor<p12bit>{ 16.f };


    template <typename O, ColorDepth CD>
    struct DenormalizePixel {
        using InputType = VectorType_t<float, cn<O>>;
        using OutputType = O;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            return Cast<InputType, OutputType>::exec(input * static_cast<float>(maxDepthValue<CD>));
        }
    };

    template <typename I, ColorDepth CD>
    struct NormalizePixel {
        using InputType = I;
        using OutputType = VectorType_t<float, cn<I>>;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            return input / static_cast<float>(maxDepthValue<CD>);
        }
    };

    template <typename I, typename O, ColorDepth CD>
    struct SaturateDenormalizePixel {
        UNARY_DECL_EXEC(I, O) {
            static_assert(std::is_same_v<VBase<I>, float>, "SaturateDenormalizePixel only works with float base types.");
            const InputType saturatedFloat = SaturateFloat<InputType>::exec(input);
            return DenormalizePixel<OutputType, CD>::exec(saturatedFloat);
        }
    };

    template <typename T, ColorDepth CD>
    struct NormalizeColorRangeDepth {
        using InputType = T;
        using OutputType = T;
        using InstanceType = UnaryType;
        using Base = typename VectorTraits<T>::base;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(std::is_floating_point_v<VBase<T>>, "NormalizeColorRangeDepth only works for floating point values");
            return input * floatShiftFactor<CD>;
        }
    };

    template <PixelFormat PF, ColorRange CR, ColorPrimitives CP, bool ALPHA, typename ReturnType = YUVOutputPixelType<PF, ALPHA>>
    struct ConvertYUVToRGB {
        static constexpr ColorDepth CD = (ColorDepth)PixelFormatTraits<PF>::depth;
        using InputType = ColorDepthPixelType<CD>;
        using OutputType = ReturnType;
        using InstanceType = UnaryType;

    private:

        // Y     -> input.x
        // Cb(U) -> input.y
        // Cr(V) -> input.z
        static constexpr __device__ __forceinline__ float3 computeRGB(const InputType& pixel) {
            const M3x3Float coefficients = ccMatrix<CR, CP, YCbCr2RGB>;
            const float CSub = subCoefficients<CD>.chroma;
            const float YSub = subCoefficients<CD>.luma;
            if constexpr (CP == bt601) {
                return MxVFloat3::exec(make_<float3>(pixel.x - YSub, pixel.y - CSub, pixel.z - CSub), coefficients);
            } else {
                return MxVFloat3::exec(make_<float3>(pixel.x, pixel.y - CSub, pixel.z - CSub), coefficients);
            }
        }

        static constexpr __device__ __forceinline__ OutputType computePixel(const InputType& pixel) {
            const float3 pixelRGBFloat = computeRGB(pixel);
            if constexpr (std::is_same_v<VBase<OutputType>, float>) {
                if constexpr (ALPHA) {
                    return { pixelRGBFloat.x, pixelRGBFloat.y, pixelRGBFloat.z, (float)maxDepthValue<CD> };
                } else {
                    return pixelRGBFloat;
                }
            } else {
                const InputType pixelRGB = fk::SaturateCast<float3, InputType>::exec(pixelRGBFloat);
                if constexpr (ALPHA) {
                    return { pixelRGB.x, pixelRGB.y, pixelRGB.z, maxDepthValue<CD> };
                } else {
                    return pixelRGB;
                }
            }

        }

    public:
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            // Pixel data shifted to the right to it's color depth numerical range
            const InputType shiftedPixel = ShiftRight<InputType>::exec(input, shiftFactor<CD>);

            // Using color depth numerical range to compute the RGB pixel
            const OutputType computedPixel = computePixel(shiftedPixel);
            if constexpr (std::is_same_v<VBase<OutputType>, float>) {
                // Moving back the pixel channels to data type numerical range, either 8bit or 16bit
                return NormalizeColorRangeDepth<OutputType, CD>::exec(computedPixel);
            } else {
                // Moving back the pixel channels to data type numerical range, either 8bit or 16bit
                return ShiftLeft<OutputType>::exec(computedPixel, shiftFactor<CD>);
            }
        }
    };

    template <PixelFormat PF>
    struct ReadYUV {
        using OutputType = ColorDepthPixelType<(ColorDepth)PixelFormatTraits<PF>::depth>;
        using PixelBaseType = ColorDepthPixelBaseType<(ColorDepth)PixelFormatTraits<PF>::depth>;
        using ParamsType = RawPtr<_2D, PixelBaseType>;
        using InputType = Point;
        using InstanceType = ReadType;
        using ReadDataType = PixelBaseType;
        static constexpr bool THREAD_FUSION{ false };
        static __device__ __forceinline__ const OutputType exec(const InputType& thread, const ParamsType& params) {
            if constexpr (PF == NV12 || PF == P010 || PF == P016 || PF == P210 || PF == P216) {
                // Planar luma
                const PixelBaseType Y = *PtrAccessor<_2D>::cr_point(thread, params);

                // Packed chroma
                const PtrDims<_2D> dims = params.dims;
                using VectorType2 = VectorType_t<PixelBaseType, 2>;
                const RawPtr<_2D, VectorType2> chromaPlane{
                    (VectorType2*)((uchar*)params.data + dims.pitch * dims.height),
                    { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                const ColorSpace CS = (ColorSpace)PixelFormatTraits<PF>::space;
                const VectorType2 UV =
                    *PtrAccessor<_2D>::cr_point({ thread.x >> 1, CS == YUV420 ? thread.y >> 1 : thread.y, thread.z }, chromaPlane);

                return { Y, UV.x, UV.y };
            } else if constexpr (PF == NV21) {
                // Planar luma
                const uchar Y = *PtrAccessor<_2D>::cr_point(thread, params);

                // Packed chroma
                const PtrDims<_2D> dims = params.dims;
                const RawPtr<_2D, uchar2> chromaPlane{
                    (uchar2*)((uchar*)params.data + dims.pitch * dims.height), { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                const uchar2 VU = *PtrAccessor<_2D>::cr_point({ thread.x >> 1, thread.y >> 1, thread.z }, chromaPlane);

                return { Y, VU.y, VU.x };
            } else if constexpr (PF == Y216 || PF == Y210) {
                const PtrDims<_2D> dims = params.dims;
                const RawPtr<_2D, ushort4> image{ (ushort4*)params.data, {dims.width >> 1, dims.height, dims.pitch} };
                const ushort4 pixel = *PtrAccessor<_2D>::cr_point({ thread.x >> 1, thread.y, thread.z }, image);
                const bool isEvenThread = IsEven<uint>::exec(thread.x);

                return { isEvenThread ? pixel.x : pixel.z, pixel.y, pixel.w };
            } else if constexpr (PF == Y416) {
                // AVYU
                // We use ushort as the type, to be compatible with the rest of the cases
                const RawPtr<_2D, ushort4> readImage{ params.data, params.dims };
                const ushort4 pixel = *PtrAccessor<_2D>::cr_point(thread, params);
                return { pixel.z, pixel.w, pixel.y, pixel.x };
            }
        }
    };

    enum ColorConversionCodes {
        COLOR_BGR2BGRA = 0,
        COLOR_RGB2RGBA = COLOR_BGR2BGRA,
        COLOR_BGRA2BGR = 1,
        COLOR_RGBA2RGB = COLOR_BGRA2BGR,
        COLOR_BGR2RGBA = 2,
        COLOR_RGB2BGRA = COLOR_BGR2RGBA,
        COLOR_RGBA2BGR = 3,
        COLOR_BGRA2RGB = COLOR_RGBA2BGR,
        COLOR_BGR2RGB = 4,
        COLOR_RGB2BGR = COLOR_BGR2RGB,
        COLOR_BGRA2RGBA = 5,
        COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,
        COLOR_BGR2GRAY = 6,
        COLOR_RGB2GRAY = 7,
        COLOR_BGRA2GRAY = 10,
        COLOR_RGBA2GRAY = 11
    };

    template <ColorConversionCodes value>
    using CCC_t = E_t<ColorConversionCodes, value>;

    using SupportedCCC = TypeList<CCC_t<COLOR_BGR2BGRA>,  CCC_t<COLOR_RGB2RGBA>,
                                  CCC_t<COLOR_BGRA2BGR>,  CCC_t<COLOR_RGBA2RGB>,
                                  CCC_t<COLOR_BGR2RGBA>,  CCC_t<COLOR_RGB2BGRA>,
                                  CCC_t<COLOR_BGRA2RGB>,  CCC_t<COLOR_RGBA2BGR>,
                                  CCC_t<COLOR_BGR2RGB>,   CCC_t<COLOR_RGB2BGR>,
                                  CCC_t<COLOR_BGRA2RGBA>, CCC_t<COLOR_RGBA2BGRA>,
                                  CCC_t<COLOR_RGB2GRAY>,  CCC_t<COLOR_RGBA2GRAY>,
                                  CCC_t<COLOR_BGR2GRAY>,  CCC_t<COLOR_BGRA2GRAY>>;

    template <ColorConversionCodes CODE>
    static constexpr bool isSuportedCCC = one_of_v<CCC_t<CODE>, SupportedCCC>;

    template <ColorConversionCodes CODE, typename I, typename O>
    struct ColorConversionType{
        static_assert(isSuportedCCC<CODE>, "Color conversion code not supported");
    };

    // Will work for COLOR_RGB2RGBA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2BGRA, I, O> {
        using type = Unary<AddAlpha<I>>;
    };

    // Will work for COLOR_RGBA2RGB too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2BGR, I, O> {
        using type = Unary<Discard<I, VectorType_t<VBase<I>, 3>>>;
    };

    // Will work for COLOR_RGB2BGRA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2RGBA, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>, AddAlpha<I>>;
    };

    // Will work for COLOR_RGBA2BGR too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2RGB, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>,
                           Discard<I, VectorType_t<VBase<I>, 3>>>;
    };

    // Will work for COLOR_RGB2BGR too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2RGB, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>>;
    };

    // Will work for COLOR_RGBA2BGRA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2RGBA, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_RGB2GRAY, I, O> {
        using type = Unary<RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2GRAY, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>, RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_RGBA2GRAY, I, O> {
        using type = Unary<RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2GRAY, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>, RGB2Gray<I, O>>;
    };

    template <ColorConversionCodes code, typename I, typename O = I>
    using ColorConversion = typename ColorConversionType<code, I, O>::type;
}; // namespace fk (Fused Kernel)
