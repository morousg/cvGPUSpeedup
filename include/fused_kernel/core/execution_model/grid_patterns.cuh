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

#include <cooperative_groups.h>

namespace cooperative_groups {};
namespace cg = cooperative_groups;

#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>

namespace fk { // namespace FusedKernel
    struct TransformGridPattern {
        private:
            template <typename T, typename DeviceFunction, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const DeviceFunction& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
                if constexpr (DeviceFunction::template is<BinaryType>) {
                    return operate(thread, DeviceFunction::Operation::exec(i_data, df.head), deviceFunctionInstances...);
                } else if constexpr (DeviceFunction::template is<UnaryType>) {
                    return operate(thread, DeviceFunction::Operation::exec(i_data), deviceFunctionInstances...);
                } else if constexpr (DeviceFunction::template is<MidWriteType>) {
                    DeviceFunction::Operation::exec(thread, i_data, df.params);
                    return operate(thread, i_data, deviceFunctionInstances...);
                } else if constexpr (DeviceFunction::template is<WriteType>) {
                    return i_data;
                }
            }

        public:
            template <typename ReadDeviceFunction, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE void exec(const ReadDeviceFunction& readDeviceFunction, const DeviceFunctionTypes&... deviceFunctionInstances) {
                const auto writeDeviceFunction = last(deviceFunctionInstances...);
                using WriteOperation = typename LastType_t<DeviceFunctionTypes...>::Operation;

                const cg::thread_block g = cg::this_thread_block();

                const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
                const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
                const uint z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
                const Point thread{ x, y, z };

                if (x < readDeviceFunction.activeThreads.x && y < readDeviceFunction.activeThreads.y) {
                    const auto tempI = ReadDeviceFunction::Operation::exec(thread, readDeviceFunction.head);
                    using ReadThreadFusion = typename ReadDeviceFunction::Operation::ThreadFusion;
                    using WriteThreadFusion = typename WriteOperation::ThreadFusion;
                    static_assert(ReadThreadFusion::times_bigger == WriteThreadFusion::times_bigger,
                        "Different Thread fusion configurations for Read and Write not supported");
                    constexpr uint elems_per_thread = ReadThreadFusion::times_bigger;
                    using BigType = typename WriteThreadFusion::type;
                    if constexpr (elems_per_thread == 1) {
                        if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                            const auto tempO = operate(thread, tempI, deviceFunctionInstances...);
                            WriteOperation::exec(thread, tempO, writeDeviceFunction.params);
                        } else {
                            WriteOperation::exec(thread, tempI, writeDeviceFunction.params);
                        }
                    } else if constexpr (elems_per_thread == 2) {
                        const auto tempI0 = ReadThreadFusion::get<0>(tempI);
                        const auto tempI1 = ReadThreadFusion::get<1>(tempI);
                        if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                            const auto tempO0 = operate(thread, tempI0, deviceFunctionInstances...);
                            const auto tempO1 = operate(thread, tempI1, deviceFunctionInstances...);
                            const VBase<BigType> tempO0_ = *((VBase<BigType>*) & tempO0);
                            const VBase<BigType> tempO1_ = *((VBase<BigType>*) & tempO1);
                            WriteOperation::exec(thread, make_<BigType>(tempO0_, tempO1_), writeDeviceFunction.params);
                        } else {
                            const VBase<BigType> tempI0_ = *((VBase<BigType>*) & tempI0);
                            const VBase<BigType> tempI1_ = *((VBase<BigType>*) & tempI1);
                            WriteOperation::exec(thread, make_<BigType>(tempI0_, tempI1_), writeDeviceFunction.params);
                        }
                    } else if constexpr (elems_per_thread == 4) {
                        const auto tempI0 = ReadThreadFusion::get<0>(tempI);
                        const auto tempI1 = ReadThreadFusion::get<1>(tempI);
                        const auto tempI2 = ReadThreadFusion::get<2>(tempI);
                        const auto tempI3 = ReadThreadFusion::get<3>(tempI);
                        if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                            const auto tempO0 = operate(thread, tempI0, deviceFunctionInstances...);
                            const auto tempO1 = operate(thread, tempI1, deviceFunctionInstances...);
                            const auto tempO2 = operate(thread, tempI2, deviceFunctionInstances...);
                            const auto tempO3 = operate(thread, tempI3, deviceFunctionInstances...);
                            const VBase<BigType> tempO0_ = *((VBase<BigType>*) & tempO0);
                            const VBase<BigType> tempO1_ = *((VBase<BigType>*) & tempO1);
                            const VBase<BigType> tempO2_ = *((VBase<BigType>*) & tempO2);
                            const VBase<BigType> tempO3_ = *((VBase<BigType>*) & tempO3);
                            const BigType result = make_<BigType>(tempO0_, tempO1_, tempO2_, tempO3_);
                            WriteOperation::exec(thread, result, writeDeviceFunction.params);
                        } else {
                            const VBase<BigType> tempI0_ = *((VBase<BigType>*) & tempI0);
                            const VBase<BigType> tempI1_ = *((VBase<BigType>*) & tempI1);
                            const VBase<BigType> tempI2_ = *((VBase<BigType>*) & tempI2);
                            const VBase<BigType> tempI3_ = *((VBase<BigType>*) & tempI3);
                            const BigType result = make_<BigType>(tempI0_, tempI1_, tempI2_, tempI3_);
                            WriteOperation::exec(thread, result, writeDeviceFunction.params);
                        }
                    }
                }
            }
    };

    template <typename SequenceSelector>
    struct DivergentBatchTransformGridPattern {
        private:
            template <int OpSequenceNumber, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const DeviceFunctionSequence<DeviceFunctionTypes...>& dfSeq,
                                                  const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
                if (OpSequenceNumber == SequenceSelector::at(z)) {
                    fk::apply(TransformGridPattern::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
                } else if constexpr (sizeof...(dfSequenceInstances) > 0) {
                    divergent_operate<OpSequenceNumber + 1>(z, dfSequenceInstances...);
                }
            }
        public:
            template <typename... DeviceFunctionSequenceTypes>
            FK_DEVICE_FUSE void exec(const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
                const cg::thread_block g = cg::this_thread_block();
                const uint z = g.group_index().z;
                divergent_operate<1>(z, dfSequenceInstances...);
            }
    };

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern::exec(deviceFunctionInstances...);
    }
    template <typename SequenceSelector, typename... DeviceFunctionSequenceTypes>
    __global__ void cuda_transform_divergent_batch(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern<SequenceSelector>::exec(dfSequenceInstances...);
    }

/*  Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
    Copyright 2023 Mediaproduccion S.L.U. (David del Rio Astorga)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License. */

    template <int BATCH>
    struct DivergentBatchTransformGridPattern_vec {
        private:
            template <int OpSequenceNumber, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const Array<int, BATCH>& dfSeqSelector,
                                                  const DeviceFunctionSequence<DeviceFunctionTypes...>& dfSeq) {
                // If the threads with this z, arrived here, we assume they have to execute this operation sequence
                fk::apply(TransformGridPattern::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
            }

            template <int OpSequenceNumber, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const Array<int, BATCH>& dfSeqSelector,
                                                  const DeviceFunctionSequence<DeviceFunctionTypes...>& dfSeq,
                                                  const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
                if (OpSequenceNumber == dfSeqSelector.at[z]) {
                    fk::apply(TransformGridPattern::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
                } else {
                    DivergentBatchTransformGridPattern_vec<BATCH>::divergent_operate<OpSequenceNumber + 1>(z, dfSeqSelector, dfSequenceInstances...);
                }
            }
        public:
            template <typename... DeviceFunctionSequenceTypes>
            FK_DEVICE_FUSE void exec(const Array<int, BATCH>& dfSeqSelector, const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
                const cg::thread_block g = cg::this_thread_block();
                const uint z = g.group_index().z;
                DivergentBatchTransformGridPattern_vec<BATCH>::divergent_operate<1>(z, dfSeqSelector, dfSequenceInstances...);
            }
    };

    template <int BATCH, typename... DeviceFunctionSequenceTypes>
    __global__ void cuda_transform_divergent_batch(const Array<int, BATCH> dfSeqSelector, const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern_vec<BATCH>::exec(dfSeqSelector, dfSequenceInstances...);
    }
}
