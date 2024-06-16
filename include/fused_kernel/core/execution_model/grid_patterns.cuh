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

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION>
    struct TransformGridPattern {
        private:
            template <typename T, typename DeviceFunction, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const DeviceFunction& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
                if constexpr (DeviceFunction::template is<WriteType>) {
                    return i_data;
                } else if constexpr (DeviceFunction::template is<MidWriteType>) {
                    DeviceFunction::Operation::exec(thread, i_data, df.params);
                    return i_data;
                } else {
                    return operate(thread, compute(thread, i_data, df), deviceFunctionInstances...);
                }
            }

            template <uint IDX, typename TFI, typename InputType, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_idx(const Point& thread, const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
                return operate(thread, TFI::get<IDX>(input), deviceFunctionInstances...);
            }

            template <typename TFI, typename InputType, uint... IDX, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE auto operate_thread_fusion_impl(std::integer_sequence<uint, IDX...> idx, const Point& thread,
                                                           const InputType& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
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

            template <typename ReadDeviceFunction, typename TFI>
            FK_DEVICE_FUSE auto read(const Point& thread, const ReadDeviceFunction& readDF) {
                if constexpr (ReadDeviceFunction::template is<ReadBackType>) {
                    if constexpr (TFI::ENABLED) {
                        return ReadDeviceFunction::Operation::exec<TFI::elems_per_thread>(thread, readDF.params, readDF.back_function);
                    } else {
                        return ReadDeviceFunction::Operation::exec(thread, readDF.params, readDF.back_function);
                    }
                } else if constexpr (ReadDeviceFunction::template is<ReadType>) {
                    if constexpr (TFI::ENABLED) {
                        return ReadDeviceFunction::Operation::exec<TFI::elems_per_thread>(thread, readDF.params);
                    } else {
                        return ReadDeviceFunction::Operation::exec(thread, readDF.params);
                    }
                }
            }

            template <typename TFI, typename ReadDeviceFunction, typename... DeviceFunctions>
            FK_DEVICE_FUSE void execute_device_functions(const Point& thread, const ReadDeviceFunction& readDF,
                                                       const DeviceFunctions&... deviceFunctionInstances) {
                using ReadOperation = typename ReadDeviceFunction::Operation;
                using WriteOperation = typename LastType_t<DeviceFunctions...>::Operation;

                const auto writeDF = ppLast(deviceFunctionInstances...);

                if constexpr (TFI::ENABLED) {
                    const auto tempI = read<ReadDeviceFunction, TFI>(thread, readDF);
                    if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                        const auto tempO = operate_thread_fusion<TFI>(thread, tempI, deviceFunctionInstances...);
                        WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDF.params);
                    } else {
                        WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDF.params);
                    }
                } else {
                    const auto tempI = read<ReadDeviceFunction, TFI>(thread, readDF);
                    if constexpr (sizeof...(deviceFunctionInstances) > 1) {
                        const auto tempO = operate(thread, tempI, deviceFunctionInstances...);
                        WriteOperation::exec(thread, tempO, writeDF.params);
                    } else {
                        WriteOperation::exec(thread, tempI, writeDF.params);
                    }
                }
            }

        public:
            template <typename ReadDeviceFunction, typename... DeviceFunctionTypes>
            FK_DEVICE_FUSE void exec(const ReadDeviceFunction& readDeviceFunction, const DeviceFunctionTypes&... deviceFunctionInstances) {
                const cg::thread_block g = cg::this_thread_block();

                const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
                const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
                const uint z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
                const Point thread{ x, y, z };

                using ReadOperation = typename ReadDeviceFunction::Operation;
                using WriteOperation = typename LastType_t<DeviceFunctionTypes...>::Operation;
                using ReadIT = typename ReadOperation::ReadDataType;
                using WriteOT = typename WriteOperation::WriteDataType;
                using TFI = ThreadFusionInfo<ReadIT, WriteOT,
                                             and_v<ReadOperation::THREAD_FUSION, WriteOperation::THREAD_FUSION, THREAD_FUSION>>;

                if (x < readDeviceFunction.activeThreads.x && y < readDeviceFunction.activeThreads.y) {
                    if constexpr (THREAD_DIVISIBLE || !TFI::ENABLED) {
                        execute_device_functions<TFI>(thread, readDeviceFunction, deviceFunctionInstances...);
                    } else {
                        if (x < readDeviceFunction.activeThreads.x - 1) {
                            execute_device_functions<TFI>(thread, readDeviceFunction, deviceFunctionInstances...);
                        } else if (x == readDeviceFunction.activeThreads.x - 1) {
                            const uint initialX = x * TFI::elems_per_thread;
                            const uint finalX = ReadOperation::num_elems_x(thread, readDeviceFunction.params);
                            uint currentX = initialX;
                            while (currentX < finalX) {
                                const Point currentThread{ currentX , thread.y, thread.z };
                                using DisabledTFI = ThreadFusionInfo<ReadIT, WriteOT, false>;
                                execute_device_functions<DisabledTFI>(currentThread, readDeviceFunction, deviceFunctionInstances...);
                                currentX++;
                            }
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

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionSequence<DeviceFunctionTypes...>& deviceFunctionInstances) {
        fk::apply(TransformGridPattern<THREAD_DIVISIBLE, THREAD_FUSION>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<true, false>::exec(deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionSequence<DeviceFunctionTypes...>& deviceFunctionInstances) {
        fk::apply(TransformGridPattern<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<true, false>::exec(deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformGridPattern<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename SequenceSelector, typename... DeviceFunctionSequenceTypes>
    __global__ void cuda_transform_divergent_batch(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern<SequenceSelector>::exec(dfSequenceInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void  __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_bounds(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<true, false>::exec(deviceFunctionInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_bounds(const DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformGridPattern<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_grid_const_bounds(const __grid_constant__ DeviceFunctionTypes... deviceFunctionInstances) {
        TransformGridPattern<true, false>::exec(deviceFunctionInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_grid_const_bounds(const __grid_constant__ DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformGridPattern<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename SequenceSelector, typename... DeviceFunctionSequenceTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_divergent_batch_bounds(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern<SequenceSelector>::exec(dfSequenceInstances...);
    }
} // namespace fk