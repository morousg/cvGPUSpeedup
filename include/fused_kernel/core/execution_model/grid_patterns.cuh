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
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION=false>
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

            template <uint IDX, typename TFI, typename InputType, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_idx(const Point& thread, const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
                return operate(thread, TFI::get<IDX>(input), deviceFunctionInstances...);
            }

            template <typename TFI, typename InputType, uint... IDX, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_thread_fusion_impl(std::integer_sequence<uint, IDX...> idx, const Point& thread, const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
                return TFI::make(operate_idx<IDX, TFI>(thread, input, deviceFunctionInstances...)...);
            }

            template <typename TFI, typename InputType, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_thread_fusion(const Point& thread, const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
                if constexpr (TFI::elems_per_thread == 1) {
                    return operate(thread, input, deviceFunctionInstances...);
                } else {
                    return operate_thread_fusion_impl<TFI>(std::make_integer_sequence<uint, TFI::elems_per_thread>(), thread, input, deviceFunctionInstances...);
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

                using ReadIT = typename ReadDeviceFunction::Operation::ReadDataType;
                using WriteOT = typename WriteOperation::WriteDataType;
                constexpr bool TF_ENABLED = ReadDeviceFunction::Operation::THREAD_FUSION && WriteOperation::THREAD_FUSION && THREAD_FUSION;
                using TFI = ThreadFusionInfo<ReadIT, WriteOT, TF_ENABLED>;

                if (x < readDeviceFunction.activeThreads.x && y < readDeviceFunction.activeThreads.y) {
                    if constexpr (TFI::ENABLED) {
                        if constexpr (THREAD_DIVISIBLE) {
                            const auto tempI = ReadDeviceFunction::Operation::exec<TFI::elems_per_thread>(thread, readDeviceFunction.params);
                            if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                                const auto tempO = operate_thread_fusion<TFI>(thread, tempI, deviceFunctionInstances...);
                                WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDeviceFunction.params);
                            } else {
                                WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDeviceFunction.params);
                            }
                        } else {
                            if (x < readDeviceFunction.activeThreads.x - 1) {
                                const auto tempI = ReadDeviceFunction::Operation::exec<TFI::elems_per_thread>(thread, readDeviceFunction.params);
                                if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                                    const auto tempO = operate_thread_fusion<TFI>(thread, tempI, deviceFunctionInstances...);
                                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDeviceFunction.params);
                                } else {
                                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDeviceFunction.params);
                                }
                            } else if (x == readDeviceFunction.activeThreads.x - 1) {
                                const uint initialX = x * TFI::elems_per_thread;
                                const uint finalX = ReadDeviceFunction::Operation::num_elems_x(thread, readDeviceFunction.params);
                                uint currentX = initialX;
                                while (currentX < finalX) {
                                    const Point currentThread{ currentX , thread.y, thread.z };
                                    const auto tempI = ReadDeviceFunction::Operation::exec(currentThread, readDeviceFunction.params);
                                    if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                                        const auto tempO = operate(currentThread, tempI, deviceFunctionInstances...);
                                        WriteOperation::exec(currentThread, tempO, writeDeviceFunction.params);
                                    } else {
                                        WriteOperation::exec(currentThread, tempI, writeDeviceFunction.params);
                                    }
                                    currentX++;
                                }
                            }
                        }
                    } else {
                        const auto tempI = ReadDeviceFunction::Operation::exec(thread, readDeviceFunction.params);
                        if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                            const auto tempO = operate(thread, tempI, deviceFunctionInstances...);
                            WriteOperation::exec(thread, tempO, writeDeviceFunction.params);
                        } else {
                            WriteOperation::exec(thread, tempI, writeDeviceFunction.params);
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
                    fk::apply(TransformGridPattern<true, false>::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
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

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<THREAD_DIVISIBLE, THREAD_FUSION>::exec(deviceFunctionInstances...);
    }
    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<false, false>::exec(deviceFunctionInstances...);
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
                fk::apply(TransformGridPattern<true>::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
            }

            template <int OpSequenceNumber, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const Array<int, BATCH>& dfSeqSelector,
                                                  const DeviceFunctionSequence<DeviceFunctionTypes...>& dfSeq,
                                                  const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
                if (OpSequenceNumber == dfSeqSelector.at[z]) {
                    fk::apply(TransformGridPattern<true>::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
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
