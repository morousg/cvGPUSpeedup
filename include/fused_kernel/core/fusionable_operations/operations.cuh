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

struct ReadType {};
struct WriteType {};
struct UnaryType {};
struct BinaryType {};
struct MidWriteType {};
struct ComposedType {};

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
template <typename T, typename P, ShiftDirection SD>
using Shift = BinaryV<Shift_<VBase<T>, SD>, T, P>;
template <typename T, typename P = uint>
using ShiftLeft = Shift<T, P, ShiftDirection::Left>;
template <typename T, typename P = uint>
using ShiftRight = Shift<T, P, ShiftDirection::Right>;

template <typename T, typename Operation>
struct CUDAVecReduce {
    UNARY_DECL_EXEC(T, VBase<T>) {
        if constexpr (cn<T> == 2) {
            return Operation::exec(input.x, input.y);
        } else if constexpr (cn<T> == 3) {
            return Operation::exec(Operation::exec(input.x, input.y), input.z);
        } else if constexpr (cn<T> == 4) {
            return Operation::exec(Operation::exec(Operation::exec(input.x, input.y), input.z), input.w);
        }
    }
};

#undef UNARY_DECL_EXEC

struct M3x3Float {
    const float3 x;
    const float3 y;
    const float3 z;
};

struct MxVFloat3 {
    BINARY_DECL_EXEC(float3, float3, M3x3Float) {
        const float3 xOut = input * params.x;
        const float3 yOut = input * params.y;
        const float3 zOut = input * params.z;
        using Reduce = CUDAVecReduce<float3, Sum<float>>;
        return { Reduce::exec(xOut), Reduce::exec(yOut), Reduce::exec(zOut) };
    }
};

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

template <ND DIMS, typename T>
struct ReadRawPtr {
    using OutputType = T;
    using InputType = Point;
    using ParamsType = RawPtr<DIMS, T>;
    using InstanceType = BinaryType;
    FK_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
        return *PtrAccessor<DIMS>::cr_point(input, params);
    }
};
template <typename T>
struct Slice2x2 {
    T _0x0;
    T _1x0;
    T _0x1;
    T _1x1;
};

template <typename ReadOperation>
struct Read2x2 {
    using ReadOutputType = typename ReadOperation::OutputType;
    using OutputType = Slice2x2<ReadOutputType>;
    using InputType = Slice2x2<Point>;
    using ParamsType = typename ReadOperation::ParamsType;
    using InstanceType = BinaryType;
    static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
        const ReadOutputType src_reg0x0 = ReadOperation::exec(input._0x0, params);
        const ReadOutputType src_reg1x0 = ReadOperation::exec(input._1x0, params);
        const ReadOutputType src_reg0x1 = ReadOperation::exec(input._0x1, params);
        const ReadOutputType src_reg1x1 = ReadOperation::exec(input._1x1, params);
        return { src_reg0x0, src_reg1x0, src_reg0x1, src_reg1x1 };
    }
};

enum InterpolationType {
    // bilinear interpolation
    INTER_LINEAR = 1,
    NONE = 17
};

template <typename PixelReadOp, InterpolationType INTER_T>
struct Interpolate;

template <typename PixelReadOp>
struct Interpolate<PixelReadOp, InterpolationType::INTER_LINEAR> {
    using OutputType = VectorType_t<float, cn<typename PixelReadOp::OutputType>>;
    using InputType = float2;
    using ParamsType = typename PixelReadOp::ParamsType;
    using InstanceType = BinaryType;
    static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
        const float src_x = input.x;
        const float src_y = input.y;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

        const int x2_read = Min<int>::exec(x2, getSourceWidth(params) - 1);
        const int y2_read = Min<int>::exec(y2, getSourceHeight(params) - 1);

        const Slice2x2<Point> readPoints{ Point(x1, y1),
                                          Point(x2_read, y1),
                                          Point(x1, y2_read),
                                          Point(x2_read, y2_read) };
 
        const auto pixels = Read2x2<PixelReadOp>::exec(readPoints, params);

        const OutputType out = (pixels._0x0 * ((x2 - src_x) * (y2 - src_y))) +
                               (pixels._1x0 * ((src_x - x1) * (y2 - src_y))) +
                               (pixels._0x1 * ((x2 - src_x) * (src_y - y1))) +
                               (pixels._1x1 * ((src_x - x1) * (src_y - y1)));
        return out;
    }
private:
    template <typename T>
    static constexpr __device__ __forceinline__ uint getSourceWidth(const RawPtr<_2D, T>& params) {
        return params.dims.width;
    }
    template <typename T>
    static constexpr __device__ __forceinline__ uint getSourceHeight(const RawPtr<_2D, T>& params) {
        return params.dims.height;
    }
    template <typename... Operations>
    static constexpr __device__ __forceinline__ uint getSourceWidth(const BinaryParams<Operations...>& head) {
        return head.params.dims.width;
    }
    template <typename... Operations>
    static constexpr __device__ __forceinline__ uint getSourceHeight(const BinaryParams<Operations...>& head) {
        return head.params.dims.height;
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

enum ColorConversionDir { YCbCr2RGB, RGB2YCbCr };

template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CCD>
constexpr M3x3Float ccMatrix{};
// Source: https://en.wikipedia.org/wiki/YCbCr
template <> constexpr M3x3Float ccMatrix<Full, bt601, YCbCr2RGB>{
    {  1.164383562f,  0.f,                1.596026786f      },
    {  1.164383562f, -0.39176229f,       -0.812967647f      },
    {  1.164383562f,  2.017232143f,       0.f               }};

// Source: https://en.wikipedia.org/wiki/YCbCr
template <> constexpr M3x3Float ccMatrix<Full, bt709, YCbCr2RGB>{
    {  1.f,           0.f,                1.5748f           },
    {  1.f,          -0.1873f,           -0.4681f           },
    {  1.f,           1.8556f,            0.f               }};

// To be verified
template <> constexpr M3x3Float ccMatrix<Limited, bt709, YCbCr2RGB>{
    {  1.f,           0.f,                1.4746f           },
    {  1.f,          -0.1646f,           -0.5713f           },
    {  1.f,           1.8814f,            0.f               }};

// Source: https://en.wikipedia.org/wiki/YCbCr
template <> constexpr M3x3Float ccMatrix<Full, bt709, RGB2YCbCr>{
    {  0.2126f,       0.7152f,            0.0722f           },
    { -0.1146f,      -0.3854f,            0.5f              },
    {  0.5f,         -0.4542f,           -0.0458f           }};

// Source: https://en.wikipedia.org/wiki/YCbCr
template <> constexpr M3x3Float ccMatrix<Full, bt2020, YCbCr2RGB>{
    {  1.f,           0.f,                1.4746f           },
    {  1.f,          -0.16455312684366f, -0.57135312684366f },
    {  1.f,           1.8814f,            0.f               }};

// Computed from ccMatrix<Full, bt2020, YCbCr2RGB>
template <> constexpr M3x3Float ccMatrix<Full, bt2020, RGB2YCbCr>{
    { -0.73792134831461f,  1.90449438202248f, -0.16657303370787f },
    {  0.39221927730127f, -1.01227510472121f,  0.62005582741994f },
    {  1.17857137414527f, -1.29153287808387f,  0.11296150393861f }};

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
            return Max<Base>::exec(0.f, Min<Base>::exec(input, 1023.f));
        } else if constexpr (CD == p12bit) {
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

template <ColorDepth CD> constexpr ColorDepthPixelBaseType<CD> shiftFactor{};
template <> constexpr ColorDepthPixelBaseType<p8bit>  shiftFactor<p8bit>{ 0u };
template <> constexpr ColorDepthPixelBaseType<p10bit> shiftFactor<p10bit>{ 6u };
template <> constexpr ColorDepthPixelBaseType<p12bit> shiftFactor<p12bit>{ 4u };

template <ColorDepth CD>
constexpr float floatShiftFactor{};
template <> constexpr float floatShiftFactor<p8bit>{ 1.f };
template <> constexpr float floatShiftFactor<p10bit>{ 64.f };
template <> constexpr float floatShiftFactor<p12bit>{ 16.f };

template <typename T, ColorDepth CD>
struct NormalizeDepth {
    using InputType = T;
    using OutputType = T;
    using InstanceType = UnaryType;
    using Base = typename VectorTraits<T>::base;
    static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
        static_assert(std::is_floating_point_v<VBase<T>>, "NormalizeDepth only works for floating point values");
        return input / static_cast<float>(maxDepthValue<CD>);
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
        if constexpr (CP == bt601) {
            const float YSub = subCoefficients<CD>.luma;
            const float CSub = subCoefficients<CD>.chroma;
            return MxVFloat3::exec(make_<float3>(pixel.x - YSub, pixel.y - CSub, pixel.z - CSub), coefficients);
        } else {
            return MxVFloat3::exec(make_<float3>(pixel.x, pixel.y, pixel.z), coefficients);
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

template <typename I>
struct AddAlpha {
    using InputType = I;
    using OutputType = VectorType_t<VBase<I>, (cn<I> + 1)>;
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

} //namespace fk
