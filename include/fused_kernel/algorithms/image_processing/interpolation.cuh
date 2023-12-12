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

#include <fused_kernel/core/fusionable_operations/memory_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/logical.cuh>

namespace fk {
    template <typename T>
    struct Slice2x2 {
        T _0x0;
        T _1x0;
        T _0x1;
        T _1x1;
    };

    struct Rect2P {
        int x1;
        int y1;
        int x2;
        int y2;
    };

    struct InterpolationParams {
        float2 srcPoint;
        Rect2P pixelPoints;
        Slice2x2<Point> readPixelPoints;
    };

    template <typename T>
    struct InterpolationPixels {
        InterpolationParams params;
        Slice2x2<T> pixels;
    };

    enum InterpolationType {
        // bilinear interpolation
        INTER_LINEAR = 1,
        NONE = 17
    };

    template <InterpolationType INTER_T>
    struct ComputeInterpolationPoints;

    template <>
    struct ComputeInterpolationPoints<InterpolationType::INTER_LINEAR> {
        using OutputType = InterpolationParams;
        using InputType = float2;
        using ParamsType = Size;
        using InstanceType = BinaryType;
        static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            const float src_x = input.x;
            const float src_y = input.y;

            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            const Rect2P pixelCoords{x1, y1, x2, y2};

            const int x2_read = Min<int>::exec(x2, params.width - 1);
            const int y2_read = Min<int>::exec(y2, params.height - 1);

            const Slice2x2<Point> readPoints{ Point(x1, y1),
                                              Point(x2_read, y1),
                                              Point(x1, y2_read),
                                              Point(x2_read, y2_read) };

            return { {src_x, src_y}, pixelCoords, readPoints };
        }
    };

    template <typename ReadOperation, InterpolationType INTER_T>
    struct ReadInterpolationPoints {};

    template <typename ReadOperation>
    struct ReadInterpolationPoints<ReadOperation, InterpolationType::INTER_LINEAR> {
        using ReadOutputType = typename ReadOperation::OutputType;
        using OutputType = InterpolationPixels<ReadOutputType>;
        using InputType = InterpolationParams;
        using ParamsType = typename ReadOperation::ParamsType;
        using InstanceType = ReadType;
        static __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            const ReadOutputType src_reg0x0 = ReadOperation::exec(input.readPixelPoints._0x0, params);
            const ReadOutputType src_reg1x0 = ReadOperation::exec(input.readPixelPoints._1x0, params);
            const ReadOutputType src_reg0x1 = ReadOperation::exec(input.readPixelPoints._0x1, params);
            const ReadOutputType src_reg1x1 = ReadOperation::exec(input.readPixelPoints._1x1, params);
            return { input, {src_reg0x0, src_reg1x0, src_reg0x1, src_reg1x1} };
        }
    };

    template <typename I, typename O, InterpolationType INTER_T>
    struct InterpolateSlice;

    template <typename I, typename O>
    struct InterpolateSlice<InterpolationPixels<I>, O, InterpolationType::INTER_LINEAR> {
        using OutputType = O;
        using InputType = InterpolationPixels<I>;
        using InstanceType = UnaryType;
        static __device__ __forceinline__ OutputType exec(const InputType& input) {
            const float2 pixPos = input.params.srcPoint;
            const Rect2P pixPoint = input.params.pixelPoints;
            const Slice2x2<I> pixels = input.pixels;
            return (pixels._0x0 * ((pixPoint.x2 - pixPos.x) * (pixPoint.y2 - pixPos.y))) +
                   (pixels._1x0 * ((pixPos.x - pixPoint.x1) * (pixPoint.y2 - pixPos.y))) +
                   (pixels._0x1 * ((pixPoint.x2 - pixPos.x) * (pixPos.y - pixPoint.y1))) +
                   (pixels._1x1 * ((pixPos.x - pixPoint.x1) * (pixPos.y - pixPoint.y1)));
        }
    };

    /*template <typename PixelReadOp, InterpolationType INTER_T>
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
    };*/
}
