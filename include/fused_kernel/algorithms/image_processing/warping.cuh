/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_WARPING
#define FK_WARPING

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>

namespace fk {
    enum WarpType { Affine = 0, Perspective = 1 };

    template<enum WarpType WType>
    struct WarpingParameters;

    template<>
    struct WarpingParameters<Affine> {
        StaticRawPtr<StaticPtrDims2D<3, 2>, float> transformMatrix;
    };

    template<>
    struct WarpingParameters<Perspective> {
        StaticRawPtr<StaticPtrDims2D<3, 3>, float> transformMatrix;
    };

    template<typename T, enum WarpType WType, typename BackIOp>
    struct Warping {
        using OutputType = VectorType_t<float, cn<T>>;
        using ReadDataType = T;
        using ParamsType = WarpingParameters<WType>;
        using BackFunction = BackIOp;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Warping<T, WType>>;
        constexpr bool THREAD_FUSION{ false };
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {

        }
    };
} // namespace fk

#endif
