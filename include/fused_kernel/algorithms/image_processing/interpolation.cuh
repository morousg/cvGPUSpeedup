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

    template <typename ReadOperation>
    struct Read2x2 {
        using ReadOutputType = typename ReadOperation::OutputType;
        using OutputType = Slice2x2<ReadOutputType>;
        using InputType = Slice2x2<Point>;
        using ParamsType = typename ReadOperation::ParamsType;
        using InstanceType = ReadType;
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
}
