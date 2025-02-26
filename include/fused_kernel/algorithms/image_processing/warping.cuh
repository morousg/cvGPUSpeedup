/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U

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

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/algorithms/image_processing/interpolation.cuh>

#include <fused_kernel/core/execution_model/default_builders_def.h>
namespace fk {
    enum WarpType { Affine = 0, Perspective = 1 };

    template<enum WarpType WType>
    struct WarpingParameters;

    template<>
    struct WarpingParameters<Affine> {
        StaticRawPtr<StaticPtrDims2D<3, 2>, float> transformMatrix;
        Size dstSize;
    };

    template<>
    struct WarpingParameters<Perspective> {
        StaticRawPtr<StaticPtrDims2D<3, 3>, float> transformMatrix;
        Size dstSize;
    };

    template<enum WarpType WT, typename BackIOp = void>
    struct Warping {
        using ReadDataType = typename BackIOp::Operation::ReadDataType;
        using OutputType = VectorType_t<float, cn<ReadDataType>>;
        using ParamsType = WarpingParameters<WT>;
        using BackFunction = BackIOp;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Warping<WT, BackIOp>>;
        static constexpr bool THREAD_FUSION{ false };

        FK_HOST_DEVICE_FUSE float2 calcCoord(const WarpingParameters<WT>& transMat, int x, int y) {
            const auto& transMatRaw = transMat.transformMatrix.data;
            if constexpr (WT == WarpType::Perspective) {
                const float coeff = 1.0f / (transMatRaw[2][0] * x + transMatRaw[2][1] * y + transMatRaw[2][2]);

                const float xcoo = coeff * (transMatRaw[0][0] * x + transMatRaw[0][1] * y + transMatRaw[0][2]);
                const float ycoo = coeff * (transMatRaw[1][0] * x + transMatRaw[1][1] * y + transMatRaw[1][2]);

                return make_<float2>(xcoo, ycoo);
            } else {
                const float xcoo = transMatRaw[0][0] * x + transMatRaw[0][1] * y + transMatRaw[0][2];
                const float ycoo = transMatRaw[1][0] * x + transMatRaw[1][1] * y + transMatRaw[1][2];

                return make_<float2>(xcoo, ycoo);
            }
        }

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            const float2 coord = calcCoord(opData.params, static_cast<int>(thread.x), static_cast<int>(thread.y));
            const Size sourceSize(BackFunction::Operation::num_elems_x(thread, opData.backFunction),
                                  BackFunction::Operation::num_elems_y(thread, opData.backFunction));
            return Interpolate<INTER_LINEAR, BackIOp>::exec(coord, { {sourceSize}, opData.backFunction });
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

        using InstantiableType = ReadBack<Warping<WT, BackIOp>>;
        DEFAULT_BUILD
    };

    template<enum WarpType WT>
    struct Warping<WT, void> {
        using OutputType = float;
        using ReadDataType = int;
        using ParamsType = WarpingParameters<WT>;
        using BackFunction = int;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Warping<WT, void>>;
        static constexpr bool THREAD_FUSION{ false };

        using InstantiableType = Instantiable<Warping<WT, void>>;

        FK_HOST_FUSE auto build(const ParamsType& params) {
            return Instantiable<Warping<WT, void>>{{params, 0}};
        }

        template <typename BackIOp>
        FK_HOST_FUSE auto build(const BackIOp& backIOp, const InstantiableType& iOp) {
            return Instantiable<Warping<WT, BackIOp>>{ {iOp.params, backIOp} };
        }

        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
