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

            template <typename ReadThreadFusion, typename InputType, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_times(const Point& thread, const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
                using WriteThreadFusion = typename LastType_t<DeviceFunctionTypes...>::Operation::ThreadFusion;
                if constexpr (ReadThreadFusion::times_bigger == 1) {
                    return operate(thread, input, deviceFunctionInstances...);
                } else if constexpr (ReadThreadFusion::times_bigger == 2) {
                    const auto tempI0 = ReadThreadFusion::get<0>(input);
                    const auto tempI1 = ReadThreadFusion::get<1>(input);
                    const auto tempO0 = operate(thread, tempI0, deviceFunctionInstances...);
                    const auto tempO1 = operate(thread, tempI1, deviceFunctionInstances...);
                    return WriteThreadFusion::make(tempO0, tempO1);
                } else if constexpr (ReadThreadFusion::times_bigger == 4) {
                    const auto tempI0 = ReadThreadFusion::get<0>(input);
                    const auto tempI1 = ReadThreadFusion::get<1>(input);
                    const auto tempI2 = ReadThreadFusion::get<2>(input);
                    const auto tempI3 = ReadThreadFusion::get<3>(input);
                    const auto tempO0 = operate(thread, tempI0, deviceFunctionInstances...);
                    const auto tempO1 = operate(thread, tempI1, deviceFunctionInstances...);
                    const auto tempO2 = operate(thread, tempI2, deviceFunctionInstances...);
                    const auto tempO3 = operate(thread, tempI3, deviceFunctionInstances...);
                    return WriteThreadFusion::make(tempO0, tempO1, tempO2, tempO3);
                } else if constexpr (ReadThreadFusion::times_bigger == 8) {
                    const auto tempI0 = ReadThreadFusion::get<0>(input);
                    const auto tempI1 = ReadThreadFusion::get<1>(input);
                    const auto tempI2 = ReadThreadFusion::get<2>(input);
                    const auto tempI3 = ReadThreadFusion::get<3>(input);
                    const auto tempI4 = ReadThreadFusion::get<4>(input);
                    const auto tempI5 = ReadThreadFusion::get<5>(input);
                    const auto tempI6 = ReadThreadFusion::get<6>(input);
                    const auto tempI7 = ReadThreadFusion::get<7>(input);
                    const auto tempO0 = operate(thread, tempI0, deviceFunctionInstances...);
                    const auto tempO1 = operate(thread, tempI1, deviceFunctionInstances...);
                    const auto tempO2 = operate(thread, tempI2, deviceFunctionInstances...);
                    const auto tempO3 = operate(thread, tempI3, deviceFunctionInstances...);
                    const auto tempO4 = operate(thread, tempI4, deviceFunctionInstances...);
                    const auto tempO5 = operate(thread, tempI5, deviceFunctionInstances...);
                    const auto tempO6 = operate(thread, tempI6, deviceFunctionInstances...);
                    const auto tempO7 = operate(thread, tempI7, deviceFunctionInstances...);
                    return WriteThreadFusion::make(tempO0, tempO1, tempO2, tempO3,
                                                   tempO4, tempO5, tempO6, tempO7);
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

                    if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                        const auto tempO = operate_times<ReadThreadFusion>(thread, tempI, deviceFunctionInstances...);
                        WriteOperation::exec(thread, tempO, writeDeviceFunction.params);
                    } else {
                        WriteOperation::exec(thread, tempI, writeDeviceFunction.params);
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
