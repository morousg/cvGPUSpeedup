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

#ifndef FK_CROP_OP
#define FK_CROP_OP

#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/data/point.h>

#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {

    template <typename BackIOp = void>
    struct Crop {
        using InstanceType = ReadBackType;
        using OutputType = typename BackIOp::Operation::OutputType;
        using ReadDataType = typename BackIOp::Operation::OutputType;
        using ParamsType = Rect;
        using BackFunction = BackIOp;
        using OperationDataType = OperationData<Crop<BackIOp>>;
        static constexpr bool THREAD_FUSION{ false };
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& iOp) {
            const Point newThread(thread.x + iOp.params.x, thread.y + iOp.params.y);
            return BackFunction::Operation::exec(newThread, iOp.back_function);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Instantiable<Crop<BackFunction>>;
        DEFAULT_BUILD

        FK_HOST_FUSE InstantiableType build(const BackFunction& backFunction, const Rect& rect) {
            return InstantiableType{ { rect, backFunction } };
        }

        DEFAULT_READ_BATCH_BUILD
    };

    template <>
    struct Crop<void> {
        using InstanceType = ReadBackType;
        using OutputType = int;
        using ReadDataType = int;
        using ParamsType = Rect;
        using BackFunction = int;
        using OperationDataType = OperationData<Crop<void>>;
        static constexpr bool THREAD_FUSION{ false };

        using InstantiableType = Instantiable<Crop<void>>;
        DEFAULT_BUILD

        FK_HOST_FUSE auto build(const Rect& rectCrop) {
            return InstantiableType{ { rectCrop, 0 } };
        }

        template <typename RealBackIOp>
        FK_HOST_FUSE auto build(const RealBackIOp& realBIOp, const InstantiableType& iOp) {
            return Crop<RealBackIOp>::build(realBIOp, iOp.params);
        }
        DEFAULT_READ_BATCH_BUILD
    };

} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif