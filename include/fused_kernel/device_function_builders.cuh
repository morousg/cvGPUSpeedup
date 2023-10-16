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

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1 };

template <typename T, InterpolationType IType>
inline const auto resize(const RawPtr<_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
    if (dSize.width != 0 && dSize.height != 0) {
        const double cfx = static_cast<double>(dSize.width) / input.dims.width;
        const double cfy = static_cast<double>(dSize.height) / input.dims.height;
        return Read<ResizeRead<T, IType>>
        { {input, static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy)},
          { (uint)dSize.width, (uint)dSize.height }
        };
    } else {
        return Read<ResizeRead<T, IType>>
        {   { input, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy) },
            { CAROTENE_NS::internal::saturate_cast<uint>(input.dims.width * fx),
              CAROTENE_NS::internal::saturate_cast<uint>(input.dims.height * fy) }
        };
    }
}

template <typename T, InterpolationType IType, int NPtr, AspectRatio AR>
inline const auto resize(const std::array<Ptr2D<T>, NPtr>& input, const Size& dsize, const int& usedPlanes, const typename ResizeRead<T, IType>::Type& backgroundValue = fk::make_set<typename ResizeRead<T, IType>::Type>(0)) {
    using ResizeArrayIgnoreType = Read<BatchRead<ResizeRead<T, IType>, NPtr>>;
    using ResizeArrayPreserveType = Read<BatchRead<ApplyROI<ResizeRead<T, IType>, OFFSET_THREADS>, NPtr>>;
    using ResizeArrayType = TypeAt_t<AR, TypeList<ResizeArrayPreserveType, ResizeArrayIgnoreType>>;

    ResizeArrayType resizeArray;
    // dsize is the size of the destination pointer, for each image
    resizeArray.activeThreads.x = dsize.width;
    resizeArray.activeThreads.y = dsize.height;
    resizeArray.activeThreads.z = usedPlanes;

    for (int i = 0; i < usedPlanes; i++) {
        const fk::PtrDims<fk::_2D> dims = input[i].dims();

        // targetWidth and targetHeight are the dimensions for the resized image
        int targetWidth, targetHeight;
        fk::ResizeReadParams<T>* interParams;
        if constexpr (AR == PRESERVE_AR) {
            float scaleFactor = dsize.height / (float)dims.height;
            targetHeight = dsize.height;
            targetWidth = static_cast<int> (ceilf(scaleFactor * dims.width));
            if (targetWidth > dsize.width) {
                scaleFactor = dsize.width / (float)dims.width;
                targetWidth = dsize.width;
                targetHeight = static_cast<int> (ceilf(scaleFactor * dims.height));
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
        interParams->ptr = input[i];
        interParams->fx = static_cast<float>(1.0 / (static_cast<double>(targetWidth) / (double)dims.width));
        interParams->fy = static_cast<float>(1.0 / (static_cast<double>(targetHeight) / (double)dims.height));
    }

    if constexpr (AR == PRESERVE_AR) {
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
