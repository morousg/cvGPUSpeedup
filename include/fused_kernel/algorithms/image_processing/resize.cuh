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
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>
#include <fused_kernel/core/data/array.cuh>

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

    enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2 };

    template <enum InterpolationType IType, enum AspectRatio = IGNORE_AR, typename T = void>
    struct ResizeReadParams {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
    };

    template <enum InterpolationType IType, typename T>
    struct ResizeReadParams<IType, PRESERVE_AR, T> {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <enum InterpolationType IType, typename T>
    struct ResizeReadParams<IType, PRESERVE_AR_RN_EVEN, T> {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <enum InterpolationType IType, enum AspectRatio AR = AspectRatio::IGNORE_AR, typename BackFunction_ = void>
    struct ResizeRead {
        using BackFunction = BackFunction_;
        static constexpr bool THREAD_FUSION{ false };
        using InstanceType = ReadBackType;
        using OutputType = typename Interpolate<IType, BackFunction>::OutputType;
        using ParamsType = ResizeReadParams<IType, AR, std::conditional_t<AR == IGNORE_AR, void, OutputType>>;
        using ReadDataType = typename BackFunction::Operation::OutputType;

    private:
        FK_HOST_DEVICE_FUSE OutputType exec_resize(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const float fx = params.src_conv_factors.x;
            const float fy = params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of ResizeRead, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the ResizeRead implementation, and not a template variable.
            return Interpolate<IType, BackFunction>::exec(rezisePoint, params.params, back_function);
        }

        FK_HOST_FUSE std::pair<int, int> compute_target_size(const Size& srcSize, const Size& dstSize) {
            const float scaleFactor = dstSize.height / (float)srcSize.height;
            const int targetHeight = dstSize.height;
            const int targetWidth = static_cast<int> (round(scaleFactor * srcSize.width));
            if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                // We round to the next even integer smaller or equal to targetWidth
                const int targetWidthTemp = targetWidth - (targetWidth % 2);
                if (targetWidthTemp > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp2 = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (round(scaleFactorTemp * srcSize.height));
                    return std::make_pair(targetWidthTemp2, targetHeightTemp - (targetHeightTemp % 2));
                } else {
                    return std::make_pair(targetWidthTemp, targetHeight);
                }
            } else {
                if (targetWidth > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (round(scaleFactorTemp * srcSize.height));
                    return std::make_pair(targetWidthTemp, targetHeightTemp);
                } else {
                    return std::make_pair(targetWidth, targetHeight);
                }
            }
        }
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            if constexpr (AR == IGNORE_AR) {
                return exec_resize(thread, params, back_function);
            } else { // Assuming PRESERVE_AR or PRESERVE_AR_RN_EVEN
                if (thread.x >= params.x1 && thread.x <= params.x2 && thread.y >= params.y1 && thread.y <= params.y2) {
                    const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                    return exec_resize(roiThread, params, back_function);
                } else {
                    return params.defaultValue;
                }
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            return params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            return params.dstSize.height;
        }
 
        using InstantiableType = ReadBack<ResizeRead<IType, AR, BackFunction_>>;
        DEFAULT_READBACK_BUILD

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE
        std::enable_if_t<AR_ == IGNORE_AR, InstantiableType>
        build(const BackFunction& backFunction, const Size& dstSize) {
            const Size srcSize = Num_elems<BackFunction>::size(Point(), backFunction);
            const double cfx = static_cast<double>(dstSize.width) / static_cast<double>(srcSize.width);
            const double cfy = static_cast<double>(dstSize.height) / static_cast<double>(srcSize.height);
            const ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize }
            };

            return { {resizeParams, backFunction} };
        }

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE
        std::enable_if_t<AR_ != IGNORE_AR, InstantiableType>
        build(const BackFunction& backFunction, const Size& dstSize, const OutputType& backgroundValue) {
            const Size srcSize = Num_elems<BackFunction>::size(Point(), backFunction);

            const auto [targetWidth, targetHeight] = compute_target_size(srcSize, dstSize);

            const int x1 = static_cast<int>((dstSize.width - targetWidth) / 2);
            const int y1 = static_cast<int>((dstSize.height - targetHeight) / 2);

            const double cfx = static_cast<double>(targetWidth) / srcSize.width;
            const double cfy = static_cast<double>(targetHeight) / srcSize.height;

            const ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize },
                /*x1*/ x1,
                /*y1*/ y1,
                /*x2*/ x1 + targetWidth - 1,
                /*y2*/ y1 + targetHeight - 1,
                /*defaultValue*/ backgroundValue
            };

            return { {resizeParams, backFunction} };
        }

        template <typename BF = BackFunction_>
        FK_HOST_FUSE
        std::enable_if_t<std::is_same_v<BF, Read<PerThreadRead<_2D, ReadDataType>>>, InstantiableType>
        build(const RawPtr<_2D, ReadDataType>& input, const Size& dSize, const double& fx, const double& fy) {
            if (dSize.width != 0 && dSize.height != 0) {
                return build(BF{ {input} }, dSize);
            } else {
                const Size computedDSize{ SaturateCast<double, int>::exec(input.dims.width * fx),
                                          SaturateCast<double, int>::exec(input.dims.height * fy) };

                return build(BF{ {input} }, computedDSize);
            }
        }

        using InstantiableSourceType = SourceReadBack<ResizeRead<IType, AR, BackFunction_>>;
        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == IGNORE_AR, InstantiableSourceType>
        build_source(const BackFunction_& backFunction, const Size& dstSize) {
            const ActiveThreads activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
            return make_source(build(backFunction, dstSize), activeThreads);
        }

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, InstantiableSourceType>
        build_source(const BackFunction_& backFunction, const Size& dstSize, const OutputType& backgroundValue) {
            const ActiveThreads activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
            return make_source(build(backFunction, dstSize, backgroundValue), activeThreads);
        }

        template <typename BF = BackFunction_>
        FK_HOST_FUSE
        std::enable_if_t<std::is_same_v<BF, Read<PerThreadRead<_2D, ReadDataType>>>, InstantiableSourceType>
        build_source(const RawPtr<_2D, ReadDataType>& input, const Size& dSize, const double& fx, const double& fy) {
            if (dSize.width != 0 && dSize.height != 0) {
                const ActiveThreads activeThreads{ static_cast<uint>(dSize.width), static_cast<uint>(dSize.height) };
                return make_source(build(input, dSize, fx, fy), activeThreads);
            } else {
                const Size computedDSize{ SaturateCast<double, int>::exec(input.dims.width * fx),
                                          SaturateCast<double, int>::exec(input.dims.height * fy) };
                const ActiveThreads activeThreads{ static_cast<uint>(computedDSize.width), static_cast<uint>(computedDSize.height) };
                return make_source(build(input, dSize, fx, fy), activeThreads);
            }
        }
    };

    template <enum InterpolationType IType, enum AspectRatio AR>
    struct ResizeRead<IType, AR, void> {
        template <typename Operation>
        FK_HOST_FUSE
        auto build(const typename ResizeRead<IType, AR, Read<Operation>>::ParamsType& params,
                   const Read<Operation>& backfunction) {
            return ResizeRead<IType, AR, Read<Operation>>::build(params, backfunction);
        }

        template <typename Operation>
        FK_HOST_FUSE
            auto build(const typename ResizeRead<IType, AR, ReadBack<Operation>>::ParamsType& params,
                const ReadBack<Operation>& backfunction) {
            return ResizeRead<IType, AR, ReadBack<Operation>>::build(params, backfunction);
        }
        
        template <typename BF, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == IGNORE_AR, ReadBack<ResizeRead<IType, AR_, BF>>>
        build(const BF& backFunction, const Size& dstSize) {
            return ResizeRead<IType, AR_, BF>::build(backFunction, dstSize);
        }

        template <typename BF, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, ReadBack<ResizeRead<IType, AR_, BF>>>
        build(const BF& backFunction, const Size& dstSize,
              const typename ResizeRead<IType, AR_, BF>::OutputType& backgroundValue) {
            return ResizeRead<IType, AR_, BF>::build(backFunction, dstSize, backgroundValue);
        }

        template <typename BF, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == IGNORE_AR, SourceReadBack<ResizeRead<IType, AR_, BF>>>
        build_source(const BF& backFunction, const Size& dstSize) {
            return ResizeRead<IType, AR_, BF>::build_source(backFunction, dstSize);
        }

        template <typename BF, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, SourceReadBack<ResizeRead<IType, AR_, BF>>>
        build_source(const BF& backFunction, const Size& dstSize,
                     const typename ResizeRead<IType, AR_, BF>::OutputType& backgroundValue) {
            return ResizeRead<IType, AR_, BF>::build_source(backFunction, dstSize, backgroundValue);
        }

        template <typename T>
        FK_HOST_FUSE
        auto build(const RawPtr<_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
            return ResizeRead<IType, AR, Instantiable<PerThreadRead<_2D, T>>>::build(input, dSize, fx, fy);
        }
        template <typename T>
        FK_HOST_FUSE
            auto build_source(const RawPtr<_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
            return ResizeRead<IType, AR, Instantiable<PerThreadRead<_2D, T>>>::build_source(input, dSize, fx, fy);
        }
    };
}; // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
