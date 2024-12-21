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

#ifndef FK_INTERPOLATION
#define FK_INTERPOLATION

#include <fused_kernel/algorithms/basic_ops/logical.cuh>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    template <typename T>
    struct Slice2x2 {
        T _0x0;
        T _1x0;
        T _0x1;
        T _1x1;
    };

    enum InterpolationType {
        INTER_LINEAR = 1,
        NONE = 17
    };

    template <InterpolationType INTER_T>
    struct InterpolationParameters {};

    template <>
    struct InterpolationParameters<InterpolationType::INTER_LINEAR> {
        Size src_size;
    };

    template <InterpolationType INTER_T, typename BackFunction_ = void>
    struct Interpolate {};

    template <typename BackFunction_>
    struct Interpolate<InterpolationType::INTER_LINEAR, BackFunction_> {
        using BackFunction = BackFunction_;
        using ReadOutputType = typename BackFunction::Operation::OutputType;
        using OutputType = fk::VectorType_t<float, cn<ReadOutputType>>;
        using InputType = float2;
        using ParamsType = InterpolationParameters<InterpolationType::INTER_LINEAR>;
        using InstanceType = TernaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const float src_x = input.x;
            const float src_y = input.y;

#ifdef __CUDA_ARCH__
            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
#else
            const int x1 = static_cast<int>(std::floor(x));
            const int y1 = static_cast<int>(std::floor(x));
#endif
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            const Size srcSize = params.src_size;
            const int x2_read = Min<int>::exec(x2, srcSize.width - 1);
            const int y2_read = Min<int>::exec(y2, srcSize.height - 1);

            const Slice2x2<Point> readPoints{ Point(x1, y1),
                                              Point(x2_read, y1),
                                              Point(x1, y2_read),
                                              Point(x2_read, y2_read) };

            const ReadOutputType src_reg0x0 = read(readPoints._0x0, back_function); //PixelReadOp::exec(readPoints._0x0, params);
            const ReadOutputType src_reg1x0 = read(readPoints._1x0, back_function); //PixelReadOp::exec(readPoints._1x0, params);
            const ReadOutputType src_reg0x1 = read(readPoints._0x1, back_function); //PixelReadOp::exec(readPoints._0x1, params);
            const ReadOutputType src_reg1x1 = read(readPoints._1x1, back_function); //PixelReadOp::exec(readPoints._1x1, params);

            return (src_reg0x0 * ((x2 - src_x) * (y2 - src_y))) +
                   (src_reg1x0 * ((src_x - x1) * (y2 - src_y))) +
                   (src_reg0x1 * ((x2 - src_x) * (src_y - y1))) +
                   (src_reg1x1 * ((src_x - x1) * (src_y - y1)));
        }
        using InstantiableType = Ternary<Interpolate<InterpolationType::INTER_LINEAR, BackFunction>>;
        DEFAULT_TERNARY_BUILD
    };

    template <InterpolationType INTER_T>
    struct Interpolate<INTER_T, void> {
        template <typename RealBackFunction>
        static constexpr __host__ __forceinline__
            auto build(const typename Interpolate<INTER_T, RealBackFunction>::ParamsType& params,
                       const RealBackFunction& backfunction) {
            return Interpolate<INTER_T, RealBackFunction>::build(params, backfunction);
        }
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
