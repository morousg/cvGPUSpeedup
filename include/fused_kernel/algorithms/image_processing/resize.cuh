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

#ifndef FK_RESIZE
#define FK_RESIZE

#include <fused_kernel/algorithms/image_processing/interpolation.cuh>
#include <fused_kernel/algorithms/image_processing/saturate.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    struct ComputeResizePoint {
        using ParamsType = float2;
        using InputType = Point;
        using InstanceType = BinaryType;
        using OutputType = float2;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& thread, const ParamsType& params) {
            // This is what makes the interpolation a resize operation
            const float fx = params.x;
            const float fy = params.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;

            return { src_x, src_y };
        }
        using InstantiableType = Binary<ComputeResizePoint>;
        DEFAULT_BINARY_BUILD
    };

    template <enum InterpolationType IType>
    struct ResizeReadParams {
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
    };

    

    template <enum InterpolationType IType, typename BackFunction_ = void>
    struct ResizeRead {
        using BackFunction = BackFunction_;
        using ParamsType = ResizeReadParams<IType>;
        static constexpr bool THREAD_FUSION{ false };
        using InstanceType = ReadBackType;
        using OutputType = typename Interpolate<IType, BackFunction>::OutputType;
        using ReadDataType = typename BackFunction::Operation::OutputType;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const float fx = params.src_conv_factors.x;
            const float fy = params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of ResizeRead, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the ResizeRead implementation, and not a template variable.
            return Interpolate<IType, BackFunction>::exec(rezisePoint, params.params, back_function);
        }
 
        using InstantiableType = ReadBack<ResizeRead<IType, BackFunction_>>;
        DEFAULT_READBACK_BUILD

        template <typename BF = BackFunction_>
        FK_HOST_FUSE
        std::enable_if_t<!std::is_same_v<BF, Read<PerThreadRead<_2D, ReadDataType>>>, InstantiableType>
        build(const BackFunction_& input, const Size& srcSize, const Size& dstSize) {
            if constexpr (IType == InterpolationType::INTER_LINEAR) {
                const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
                const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
                const ParamsType resizeParams{
                    { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                    { srcSize }
                };

                return { {resizeParams, input} };
            }
        }

        using InstantiableSourceType = SourceReadBack<ResizeRead<IType, BackFunction_>>;
        template <typename BF = BackFunction_>
        FK_HOST_FUSE 
        std::enable_if_t<!std::is_same_v<BF, Read<PerThreadRead<_2D, ReadDataType>>>, InstantiableSourceType>
        build_source(const BackFunction_& input, const Size& srcSize, const Size& dstSize) {
            if constexpr (IType == InterpolationType::INTER_LINEAR) {
                const dim3 activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
                const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
                const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
                const ParamsType resizeParams {
                    { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                    { srcSize }
                };

                return { {resizeParams, input}, activeThreads };
            }
        }

        template <typename BF = BackFunction_>
        FK_HOST_FUSE
        std::enable_if_t<std::is_same_v<BF, Read<PerThreadRead<_2D, ReadDataType>>>, InstantiableSourceType>
        build_source(const RawPtr<_2D, ReadDataType>& input, const Size& dSize, const double& fx, const double& fy) {
            const BackFunction backDF{ input };
            if (dSize.width != 0 && dSize.height != 0) {
                const dim3 activeThreads{ static_cast<uint>(dSize.width), static_cast<uint>(dSize.height) };
                const double cfx = static_cast<double>(dSize.width) / input.dims.width;
                const double cfy = static_cast<double>(dSize.height) / input.dims.height;
                const ParamsType resizeParams{
                    {static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy)},
                    {Size(input.dims.width, input.dims.height)}
                };

                return { {resizeParams, backDF}, activeThreads };
            } else {
                const Size computedDSize{ SaturateCast<double, int>::exec(input.dims.width * fx),
                                          SaturateCast<double, int>::exec(input.dims.height * fy) };

                const dim3 activeThreads{ static_cast<uint>(computedDSize.width), static_cast<uint>(computedDSize.height) };
                const ParamsType resizeParams{
                    {static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy)},
                    {Size(input.dims.width, input.dims.height)}
                };

                return { {resizeParams, backDF}, activeThreads };
            }
        }
    };

    template <enum InterpolationType IType>
    struct ResizeRead<IType, void> {
        using ParamsType = ResizeReadParams<IType>;
        template <typename RealBackFunction>
        FK_HOST_FUSE
        auto build(const typename ResizeRead<IType, RealBackFunction>::ParamsType& params,
                   const RealBackFunction& backfunction) {
            return ResizeRead<IType, RealBackFunction>::build(params, backfunction);
        }
        FK_HOST_FUSE
        ParamsType compute_params(const Size& srcSize, const Size& dstSize) {
            if constexpr (IType == InterpolationType::INTER_LINEAR) {
                const dim3 activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
                const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
                const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
                const ParamsType resizeParams{
                    { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                    { srcSize }
                };

                return resizeParams;
            }
        }
        template <typename T>
        ParamsType compute_params(const Ptr2D<T>& sourcePtr, const Size& dstSize) {
            const Size srcSize{ static_cast<int>(sourcePtr.dims().width),
                                static_cast<int>(sourcePtr.dims().height) };
            if constexpr (IType == InterpolationType::INTER_LINEAR) {
                const dim3 activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
                const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
                const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
                const ParamsType resizeParams{
                    { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                    { srcSize }
                };

                return resizeParams;
            }
        }
    };
}; // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
