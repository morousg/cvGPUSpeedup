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

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>
#include <fused_kernel/algorithms/builder_utils/lapack.h>

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
    struct Warping;
    
    template<typename T, typename BackIOp>
    struct Warping<T, WarpType::Perspective, BackIOp> {
        using OutputType = VectorType_t<float, cn<T>>;
        using ReadDataType = T;
        using ParamsType = WarpingParameters<WarpType::Perspective>;
        using BackFunction = BackIOp;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Warping<T, WarpType::Perspective, BackIOp>>;
        static constexpr bool THREAD_FUSION{ false };
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread,
                                            const OperationDataType& opData) {

        }

        using InstantiableType = ReadBack<Warping<T, WarpType::Perspective, BackIOp>>;
        DEFAULT_BUILD

        FK_HOST_FUSE InstantiableType build(const std::array<Point_<float, _2D>, 4>& src,
                                            const std::array<Point_<float, _2D>, 4>& dst,
                                            const DecompTypes& solveMethod) {
            StaticPtr<StaticRawPtr<StaticPtrDims2D<8, 8>, float>> A{};
            StaticPtr<StaticRawPtr<StaticPtrDims2D<8, 1>, float>> B{};
            StaticPtr<StaticRawPtr<StaticPtrDims2D<8, 1>, float>> X{};

            for (int i = 0; i < 4; ++i) {
                A.ptr_a.data[i][0] = A.ptr_a.data[i + 4][3] = src[i].x;
                A.ptr_a.data[i][1] = A.ptr_a.data[i + 4][4] = src[i].y;
                A.ptr_a.data[i][2] = A.ptr_a.data[i + 4][5] = 1;
                A.ptr_a.data[i][3] =
                    A.ptr_a.data[i][4] = A.ptr_a.data[i][5] =
                    A.ptr_a.data[i + 4][0] = A.ptr_a.data[i + 4][1] =
                    A.ptr_a.data[i + 4][2] = 0;
                A.ptr_a.data[i][6] = -src[i].x * dst[i].x;
                A.ptr_a.data[i][7] = -src[i].y * dst[i].x;
                A.ptr_a.data[i + 4][6] = -src[i].x * dst[i].y;
                A.ptr_a.data[i + 4][7] = -src[i].y * dst[i].y;
                B.ptr_a.data[0][i] = dst[i].x;
                B.ptr_a.data[0][i + 4] = dst[i].y;
            }

            solve(A, B, X, solveMethod);
            decltype(X)::At::write(Point(3,3), X.ptr_a, 1.f);

            return InstantiableType{};
        }
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
