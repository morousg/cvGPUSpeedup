/* Copyright 2023-2025 Oscar Amoros Huguet

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
#include <fused_kernel/core/data/array.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    struct ComputeResizePoint {
        using ParamsType = float2;
        using InputType = Point;
        using InstanceType = BinaryType;
        using OutputType = float2;
        using OperationDataType = OperationData<ComputeResizePoint>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& thread, const OperationDataType& opData) {
            // This is what makes the interpolation a resize operation
            const float fx = opData.params.x;
            const float fy = opData.params.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;

            return { src_x, src_y };
        }
        using InstantiableType = Binary<ComputeResizePoint>;
        DEFAULT_BUILD
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
        using InstanceType = ReadBackType;
        using OutputType = typename Interpolate<IType, BackFunction>::OutputType;
        using ParamsType = ResizeReadParams<IType, AR, std::conditional_t<AR == IGNORE_AR, void, OutputType>>;
        using ReadDataType = typename BackFunction::Operation::OutputType;
        using OperationDataType = OperationData<ResizeRead<IType, AR, BackFunction>>;
        static constexpr bool THREAD_FUSION{ false };
    private:
        FK_HOST_DEVICE_FUSE OutputType exec_resize(const Point& thread, const OperationDataType& opData) {
            const float fx = opData.params.src_conv_factors.x;
            const float fy = opData.params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of ResizeRead, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the ResizeRead implementation, and not a template variable.
            // But, it would be relatively easy to change Interpolate with anything else if needed.
            return Interpolate<IType, BackFunction>::exec(rezisePoint, { opData.params.params, opData.back_function });
        }

        FK_HOST_FUSE std::pair<int, int> compute_target_size(const Size& srcSize, const Size& dstSize) {
            const float scaleFactor = dstSize.height / (float)srcSize.height;
            const int targetHeight = dstSize.height;
            const int targetWidth = static_cast<int>(cxp::round(scaleFactor * srcSize.width));
            if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                // We round to the next even integer smaller or equal to targetWidth
                const int targetWidthTemp = targetWidth - (targetWidth % 2);
                if (targetWidthTemp > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp2 = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (cxp::round(scaleFactorTemp * srcSize.height));
                    return std::make_pair(targetWidthTemp2, targetHeightTemp - (targetHeightTemp % 2));
                } else {
                    return std::make_pair(targetWidthTemp, targetHeight);
                }
            } else {
                if (targetWidth > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (cxp::round(scaleFactorTemp * srcSize.height));
                    return std::make_pair(targetWidthTemp, targetHeightTemp);
                } else {
                    return std::make_pair(targetWidth, targetHeight);
                }
            }
        }
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            if constexpr (AR == IGNORE_AR) {
                return exec_resize(thread, opData);
            } else { // Assuming PRESERVE_AR or PRESERVE_AR_RN_EVEN
                if (thread.x >= opData.params.x1 && thread.x <= opData.params.x2 &&
                    thread.y >= opData.params.y1 && thread.y <= opData.params.y2) {
                    const Point roiThread(thread.x - opData.params.x1, thread.y - opData.params.y1, thread.z);
                    return exec_resize(roiThread, opData);
                } else {
                    return opData.params.defaultValue;
                }
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }
 
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = ReadBack<ResizeRead<IType, AR, BackFunction_>>;
        DEFAULT_BUILD

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

        DEFAULT_READ_BATCH_BUILD
    };

    template <enum AspectRatio AR, typename T = void>
    struct IncompleteResizeReadParams {
        Size dstSize;
        T defaultValue;
    };

    template <enum AspectRatio AR>
    struct IncompleteResizeReadParams<AR, void> {
        Size dstSize;
    };

    template <enum InterpolationType IType, enum AspectRatio AR, typename T>
    struct ResizeRead<IType, AR, TypeList<void, T>> {
        using BackFunction = int;
        static constexpr bool THREAD_FUSION{ false };
        using InstanceType = ReadBackType;
        using OutputType = int;
        using ParamsType = IncompleteResizeReadParams<AR, T>;
        using ReadDataType = int;
        using OperationDataType = OperationData<ResizeRead<IType, AR, TypeList<void, T>>>;

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, ReadBack<ResizeRead<IType, AR_, TypeList<void, T>>>>
        build(const Size& dstSize,
              const T& backgroundValue) {
            return ReadBack<ResizeRead<IType, AR_, TypeList<void, T>>>{{{dstSize, backgroundValue}, 0}};
        }

        template <typename ReadIOp, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, ReadBack<ResizeRead<IType, AR_, ReadIOp>>>
        build(const ReadIOp& readIOp, const ReadBack<ResizeRead<IType, AR_, TypeList<void, T>>>& iOp) {
            static_assert(std::is_same_v<T, typename ReadIOp::Operation::OutputType>,
                "Background value type is not the same as the provided ReadOperation OutputType");
            return ResizeRead<IType, AR_, ReadIOp>::build(readIOp, iOp.params.dstSize, iOp.params.defaultValue);
        }
        using InstantiableType = Instantiable<ResizeRead<IType, AR, TypeList<void, T>>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <enum InterpolationType IType>
    struct ResizeRead<IType, IGNORE_AR, TypeList<void, void>> {
        using BackFunction = int;
        static constexpr bool THREAD_FUSION{ false };
        using InstanceType = ReadBackType;
        using OutputType = int;
        using ParamsType = IncompleteResizeReadParams<IGNORE_AR, void>;
        using ReadDataType = int;
        using OperationDataType = OperationData<ResizeRead<IType, IGNORE_AR, TypeList<void, void>>>;

        FK_HOST_FUSE ReadBack<ResizeRead<IType, IGNORE_AR, TypeList<void, void>>>
        build(const Size& dstSize) {
            return ReadBack<ResizeRead<IType, IGNORE_AR, TypeList<void, void>>>{{{dstSize}, 0}};
        }

        template <typename ReadIOp>
        FK_HOST_FUSE ReadBack<ResizeRead<IType, IGNORE_AR, ReadIOp>>
        build(const ReadIOp& readIOp, const ReadBack<ResizeRead<IType, IGNORE_AR, TypeList<void, void>>>& iOp) {
            return ResizeRead<IType, IGNORE_AR, ReadIOp>::build(readIOp, iOp.params.dstSize);
        }

        using InstantiableType = Instantiable<ResizeRead<IType, IGNORE_AR, TypeList<void, void>>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <enum InterpolationType IType, enum AspectRatio AR>
    struct ResizeRead<IType, AR, void> {
        template <typename Operation>
        FK_HOST_FUSE auto build(const typename ResizeRead<IType, AR, Read<Operation>>::ParamsType& params,
                                const Read<Operation>& backfunction) {
            return ResizeRead<IType, AR, Read<Operation>>::build(params, backfunction);
        }

        template <typename Operation>
        FK_HOST_FUSE auto build(const typename ResizeRead<IType, AR, ReadBack<Operation>>::ParamsType& params,
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

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == IGNORE_AR, ReadBack<ResizeRead<IType, AR_, TypeList<void, void>>>>
        build(const Size& dstSize) {
            return ResizeRead<IType, AR_, TypeList<void, void>>::build(dstSize);
        }

        template <typename T, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != IGNORE_AR, ReadBack<ResizeRead<IType, AR_, TypeList<void, T>>>>
        build(const Size& dstSize,
              const T& backgroundValue) {
            return ResizeRead<IType, AR_, TypeList<void, T>>::build(dstSize, backgroundValue);
        }

        template <typename T>
        FK_HOST_FUSE auto build(const RawPtr<_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
            return ResizeRead<IType, AR, Instantiable<PerThreadRead<_2D, T>>>::build(input, dSize, fx, fy);
        }
        DEFAULT_READ_BATCH_BUILD
    };
}; // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
