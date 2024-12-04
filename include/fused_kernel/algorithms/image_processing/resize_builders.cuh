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
#pragma once

#include <fused_kernel/core/external/carotene/saturate_cast.hpp>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/core/execution_model/memory_operation_builders.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>

namespace fk {
    enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2 };

    template <typename BackFunction, enum InterpolationType IType>
    inline const auto resize(const BackFunction& input,
        const Size& srcSize, const Size& dstSize) {
        if constexpr (IType == InterpolationType::INTER_LINEAR) {
            using ResizeDF = SourceReadBack<ResizeRead<BackFunction, IType>>;
            const dim3 activeThreads{ static_cast<uint>(dstSize.width), static_cast<uint>(dstSize.height) };
            const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
            const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
            const typename ResizeDF::Operation::ParamsType resizeParams{
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize }
            };

            const ResizeDF resizeInstance{ resizeParams, input, activeThreads };

            return resizeInstance;
        }
    }

    template <typename I, enum InterpolationType IType>
    inline const auto resize(const RawPtr<_2D, I>& input, const Size& dSize, const double& fx, const double& fy) {
        using BackFunction = Read<PerThreadRead<_2D, I>>;
        using ResizeDF = SourceReadBack<ResizeRead<BackFunction, IType>>;

        const BackFunction backDF{ input };
        if (dSize.width != 0 && dSize.height != 0) {
            const dim3 activeThreads{ static_cast<uint>(dSize.width), static_cast<uint>(dSize.height) };
            const double cfx = static_cast<double>(dSize.width) / input.dims.width;
            const double cfy = static_cast<double>(dSize.height) / input.dims.height;
            const typename ResizeDF::Operation::ParamsType resizeParams{
                {static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy)},
                {Size(input.dims.width, input.dims.height)}
            };

            const ResizeDF resizeInstance{ resizeParams, backDF, activeThreads };

            return resizeInstance;
        } else {
            const Size computedDSize{ CAROTENE_NS::internal::saturate_cast<int>(input.dims.width * fx),
                                      CAROTENE_NS::internal::saturate_cast<int>(input.dims.height * fy) };

            const dim3 activeThreads{ static_cast<uint>(computedDSize.width), static_cast<uint>(computedDSize.height) };
            const typename ResizeDF::Operation::ParamsType resizeParams{
                {static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy)},
                {Size(input.dims.width, input.dims.height)}
            };

            const ResizeDF resizeInstance{ resizeParams, backDF, activeThreads };

            return resizeInstance;
        }
    }

    template <typename InputType, enum InterpolationType IType>
    struct GetResizeReadParams {
        using OutputType = ResizeReadParams<IType>;
        template <int Idx>
        static constexpr inline OutputType transform(const int& usedPlanes, const InputType& input, const int& targetWidth, const int& targetHeight) {
            const PtrDims<_2D> sourceDims = input.dims;
            if constexpr (IType == InterpolationType::INTER_LINEAR) {
                return ResizeReadParams<IType>{ { static_cast<float>(1.0 / (static_cast<double>(targetWidth) / (double)sourceDims.width)),
                    static_cast<float>(1.0 / (static_cast<double>(targetHeight) / (double)sourceDims.height)) },
                    { Size(sourceDims.width, sourceDims.height) } };
            }
        }
    };

    template <typename InputType, AspectRatio AR, typename T, enum InterpolationType IType>
    struct GetApplyROYParams {
        using OutputType = std::pair<ApplyROIParams<T>, ResizeReadParams<IType>>;
        template <int Idx>
        static constexpr inline OutputType
            transform(const int& usedPlanes, const InputType& inputElem, const Size& dSize, const T& backgroundValue) {
            if (Idx < usedPlanes) {
                const PtrDims<_2D> sourceDims = inputElem.dims;
                float scaleFactor = dSize.height / (float)sourceDims.height;
                int targetHeight = dSize.height;
                int targetWidth = static_cast<int> (round(scaleFactor * sourceDims.width));
                if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                    // We round to the next even integer smaller or equal to targetWidth
                    targetWidth -= targetWidth % 2;
                }
                if (targetWidth > dSize.width) {
                    scaleFactor = dSize.width / (float)sourceDims.width;
                    targetWidth = dSize.width;
                    targetHeight = static_cast<int> (round(scaleFactor * sourceDims.height));
                    if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                        // We round to the next even integer smaller or equal to targetHeight
                        targetHeight -= targetHeight % 2;
                    }
                }

                const int x1 = static_cast<int>((dSize.width - targetWidth) / 2);
                const int y1 = static_cast<int>((dSize.height - targetHeight) / 2);

                return { { /*x1*/ x1,
                    /*y1*/ y1,
                    /*x2*/ x1 + targetWidth - 1,
                    /*y2*/ y1 + targetHeight - 1,
                    /*defaultValue*/ backgroundValue },
                    GetResizeReadParams<InputType, IType>::template transform<Idx>(usedPlanes, inputElem, targetWidth, targetHeight) };
            } else {
                return { { /*x1*/ -1,
                    /*y1*/ -1,
                    /*x2*/ -1,
                    /*y2*/ -1,
                    /*defaultValue*/ backgroundValue },
                    ResizeReadParams<IType>{} };
            }
        }
    };

    template <typename PixelReadOp, typename O, enum InterpolationType IType, size_t NPtr, enum AspectRatio AR>
    inline const auto resize(const std::array<typename PixelReadOp::ParamsType, NPtr>& input,
        const fk::Size& dsize, const int& usedPlanes,
        const O& backgroundValue = fk::make_set<O>(0)) {
        using ReadPixelDF = Read<PixelReadOp>;
        using ResizeReadOp = ResizeRead<ReadPixelDF, IType>;

        const std::array<ReadPixelDF, NPtr> pixelReadDFs = paramsArrayToDFArray<PixelReadOp>(input);

        if constexpr (AR != IGNORE_AR) {
            // We will instantiate this types only if we use AR
            using ResizeDF = ReadBack<ResizeReadOp>;
            using ApplyROYOp = ApplyROI<ResizeDF, ROI::OFFSET_THREADS>;

            const std::array<std::pair<ApplyROIParams<O>, ResizeReadParams<IType>>, NPtr> roiAndResizeParams =
                static_transform<GetApplyROYParams<typename PixelReadOp::ParamsType, AR, O, IType>>(usedPlanes, input, dsize, backgroundValue);
            const auto roiParams = static_transform_get_first(roiAndResizeParams);
            const auto resizeParams = static_transform_get_second(roiAndResizeParams);
            const std::array<ResizeDF, NPtr> resizeDFs = paramsArrayToDFArray<ResizeReadOp>(resizeParams, pixelReadDFs);
            return buildBatchReadDF<ApplyROYOp>(roiParams, resizeDFs, dsize);
        } else {
            const std::array<ResizeReadParams<IType>, NPtr> resizeParams =
                static_transform<GetResizeReadParams<typename PixelReadOp::ParamsType, IType>>(usedPlanes, input, dsize.width, dsize.height);
            return buildBatchReadDF<ResizeReadOp>(resizeParams, pixelReadDFs, dsize);
        }
    }
}
