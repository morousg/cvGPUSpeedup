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

#include <external/carotene/saturate_cast.hpp>
#include "device_functions.cuh"
#include "memory_operations.cuh"

namespace fk {

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2 };

template <typename PixelReadOp, InterpolationType IType>
inline const auto resize(const typename PixelReadOp::ParamsType& input, const Size& srcSize, const Size& dstSize) {
    const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
    const double cfy = static_cast<double>(dstSize.height) / srcSize.height;
    return Read<ResizeRead<PixelReadOp, IType>>
    { {{input, srcSize}, static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy)},
        { (uint)dstSize.width, (uint)dstSize.height }
    };
}

template <typename I, InterpolationType IType>
inline const auto resize(const RawPtr<_2D, I>& input, const Size& dSize, const double& fx, const double& fy) {
    const fk::Size sourceSize(input.dims.width, input.dims.height);
    if (dSize.width != 0 && dSize.height != 0) {
        const double cfx = static_cast<double>(dSize.width) / input.dims.width;
        const double cfy = static_cast<double>(dSize.height) / input.dims.height;
        return Read<ResizeRead<ReadRawPtr<_2D, I>, IType>>
        { {{input, sourceSize}, static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy)},
          { (uint)dSize.width, (uint)dSize.height }
        };
    } else {
        return Read<ResizeRead<ReadRawPtr<_2D, I>, IType>>
        {   { {input, sourceSize}, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy) },
            { CAROTENE_NS::internal::saturate_cast<uint>(input.dims.width * fx),
              CAROTENE_NS::internal::saturate_cast<uint>(input.dims.height * fy) }
        };
    }
}

template <typename PixelReadOp, InterpolationType IType, int NPtr, AspectRatio AR>
inline const auto resize(const std::array<typename PixelReadOp::ParamsType, NPtr>& input,
                         const Size& dsize, const int& usedPlanes,
                         const typename ResizeRead<PixelReadOp, IType>::OutputType& backgroundValue = fk::make_set<typename ResizeRead<PixelReadOp, IType>::OutputType>(0)) {
    using ResizeArrayIgnoreType = Read<BatchRead<ResizeRead<PixelReadOp, IType>, NPtr>>;
    using ResizeArrayPreserveType = Read<BatchRead<ApplyROI<ResizeRead<PixelReadOp, IType>, OFFSET_THREADS>, NPtr>>;
    using ResizeArrayPreserveRoundEvenType = Read<BatchRead<ApplyROI<ResizeRead<PixelReadOp, IType>, OFFSET_THREADS>, NPtr>>;
    using ResizeArrayType = TypeAt_t<AR, TypeList<ResizeArrayPreserveType, ResizeArrayIgnoreType, ResizeArrayPreserveRoundEvenType>>;

    ResizeArrayType resizeArray;
    // dsize is the size of the destination pointer, for each image
    resizeArray.activeThreads.x = dsize.width;
    resizeArray.activeThreads.y = dsize.height;
    resizeArray.activeThreads.z = usedPlanes;

    for (int i = 0; i < usedPlanes; i++) {
        const fk::PtrDims<fk::_2D> dims = input[i].dims;

        // targetWidth and targetHeight are the dimensions for the resized image
        int targetWidth, targetHeight;
        fk::ResizeReadParams<Interpolate<PixelReadOp, IType>>* interParams;
        if constexpr (AR != IGNORE_AR) {
            float scaleFactor = dsize.height / (float)dims.height;
            targetHeight = dsize.height;
            targetWidth = static_cast<int> (round(scaleFactor * dims.width));
            if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                // We round to the next even integer smaller or equal to targetWidth
                targetWidth -= targetWidth % 2;
            }
            if (targetWidth > dsize.width) {
                scaleFactor = dsize.width / (float)dims.width;
                targetWidth = dsize.width;
                targetHeight = static_cast<int> (round(scaleFactor * dims.height));
                if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                    // We round to the next even integer smaller or equal to targetHeight
                    targetHeight -= targetHeight % 2;
                }
            }
            resizeArray.activeThreads.z = NPtr;
            resizeArray.params[i].x1 = (dsize.width - targetWidth) / 2;
            resizeArray.params[i].x2 = resizeArray.params[i].x1 + targetWidth - 1;
            resizeArray.params[i].y1 = (dsize.height - targetHeight) / 2;
            resizeArray.params[i].y2 = resizeArray.params[i].y1 + targetHeight - 1;
            resizeArray.params[i].defaultValue = backgroundValue;
            interParams = &resizeArray.params[i].params;
        } else {
            targetWidth = dsize.width;
            targetHeight = dsize.height;
            interParams = &resizeArray.params[i];
        }
        interParams->params = { input[i],  Size(dims.width, dims.height) };
        interParams->fx = static_cast<float>(1.0 / (static_cast<double>(targetWidth) / (double)dims.width));
        interParams->fy = static_cast<float>(1.0 / (static_cast<double>(targetHeight) / (double)dims.height));
    }

    if constexpr (AR != IGNORE_AR) {
        for (int i = usedPlanes; i < NPtr; i++) {
            resizeArray.params[i].x1 = -1;
            resizeArray.params[i].x2 = -1;
            resizeArray.params[i].y1 = -1;
            resizeArray.params[i].y2 = -1;
            resizeArray.params[i].defaultValue = backgroundValue;
        }
    }
    return resizeArray;
}

}; // namespace fk
