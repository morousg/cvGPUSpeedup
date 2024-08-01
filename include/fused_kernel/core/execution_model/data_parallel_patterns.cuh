/* Copyright 2023-2024 Oscar Amoros Huguet

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

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION>
    struct TransformDPP {
    private:
        template <typename T, typename DeviceFunction, typename... DeviceFunctionTypes>
        FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const DeviceFunction& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
            if constexpr (DeviceFunction::template is<WriteType>) {
                return i_data;
            } else if constexpr (DeviceFunction::template is<MidWriteType>) {
                DeviceFunction::Operation::exec(thread, i_data, df.params);
                return i_data;
            } else {
                return operate(thread, compute(i_data, df), deviceFunctionInstances...);
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
    struct DivergentBatchTransformDPP {
    private:
        template <int OpSequenceNumber, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
        FK_DEVICE_FUSE void divergent_operate(const uint& z, const DeviceFunctionSequence<DeviceFunctionTypes...>& dfSeq,
            const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
            if (OpSequenceNumber == SequenceSelector::at(z)) {
                fk::apply(TransformDPP<true, false>::exec<DeviceFunctionTypes...>, dfSeq.deviceFunctions);
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

    template <uint B_x, uint B_y, bool ReturnResult, typename ReadDF, typename WriteIntermediateDF, typename ReadIntermediateDF, typename FinalDF, typename WriteDF, typename ReadFinalDF, typename... DeviceFunctionTuples>
    struct Reduce2D_DPP {
        using Instantiation = Tuple<ReadDF, WriteIntermediateDF, ReadIntermediateDF, FinalDF, WriteDF, ReadFinalDF, Tuple<DeviceFunctionTuples...>>;
        
        using OutputType = std::conditional_t<ReturnResult, Tuple<typename ReadDF::Operation::OutputType, typename FinalDF::Operation::OutputType>, void>;
    private:
        template <typename T, typename TupleType>
        FK_DEVICE_FUSE auto execute_firstDF_one_reduction(const T& new_value, const TupleType& dfTuple) {
            static_assert(TupleType::size == 1 || TupleType::size == 2, "Tuple must have size 1 or 2 in execute_firstDF");
            if constexpr (TupleType::size == 1) {
                return new_value;
            } else if constexpr (TupleType::size == 2) {
                return compute(new_value, get_v<0>(dfTuple));
            }
        }

        template <typename T, int... Idx, typename... DeviceFunctionTuples>
        FK_DEVICE_FUSE auto execute_firstDF_helper(const T& new_value,
            const std::integer_sequence<int, Idx...>& iSeq,
            const DeviceFunctionTuples&... reductions) {
            using ReturnType = Tuple<decltype(execute_firstDF_one_reduction(new_value, get_v<Idx>(reductions)))...>;
            return ReturnType{ {execute_firstDF_one_reduction(new_value, get_v<Idx>(reductions))...} };
        }

        template <typename T, typename... DeviceFunctionTuples>
        FK_DEVICE_FUSE auto execute_firstDF(const T& new_value, const DeviceFunctionTuples&... reductions) {
            return execute_firstDF_helper(new_value, std::make_integer_sequence<int, sizeof...(DeviceFunctionTuples)>{}, reductions...);
        }

        template <typename T, typename TupleType>
        FK_DEVICE_FUSE T execute_secondDF_one_reduction(const T& from_shared, const T& accum, const TupleType&) {
            static_assert(TupleType::size == 1 || TupleType::size == 2, "Tuple must have size 1 or 2 in execute_firstDF");
            constexpr int OP_IDX = TupleType::size - 1;
            using DF = get_type_t<OP_IDX, TupleType>;
            return DF::exec(Tuple<T, T>{from_shared, accum});
        }

        template <typename TupleType, int... Idx, typename... DeviceFunctionTuples>
        FK_DEVICE_FUSE TupleType execute_secondDF_helper(const TupleType& from_shared, const TupleType& accum,
            const std::integer_sequence<int, Idx...>& iSeq,
            const DeviceFunctionTuples&... reductions) {
            return TupleType{ execute_secondDF_one_reduction(get_v<Idx>(from_shared), get_v<Idx>(accum), get_v<Idx>(reductions))... };
        }

        template <typename TupleType, typename... DeviceFunctionTuples>
        FK_DEVICE_FUSE TupleType execute_secondDF(
            const TupleType& from_shared, const TupleType& accum, const DeviceFunctionTuples&... reductions) {
            return execute_secondDF_helper(from_shared, accum, std::make_integer_sequence<int, sizeof...(DeviceFunctionTuples)>{}, reductions...);
        }

    public:
        static __device__ __forceinline__ OutputType exec(const Instantiation& instantiation) {
            const 
            constexpr size_t N = sizeof...(DeviceFunctionTuples);
            using ReadType = typename ReadDF::Operation::OutputType;
            using OutputType = typename FinalDF::Operation::OutputType;

            // Assuming we have at least as many threads as num_elems to be reduced
            const cg::thread_block g = cg::this_thread_block();

            const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
            const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
            const uint z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes

            const Point global_threads{ x, y, z };

            ReadType read_value = make_set<ReadType>(0);
            using AccumType = decltype(execute_firstDF(read_value, reductions...));
            __shared__ AccumType shared_data[B_y][B_x];
            const uint2 block_threads = { g.thread_index().x, g.thread_index().y };
            // Decide which threads work
            if (x < readDF.activeThreads.x && y < readDF.activeThreads.y) {
                // Do read -> ReadDF
                read_value = ReadDF::exec(global_threads, readDF.params);
                // Each thread applies the firstDF to it’s read_value, for each reduction
                const AccumType accum = execute_firstDF(read_value, reductions...);
                // Optimizable with warp reduction
                // Write TB to shared memory
                shared_data[block_threads.y][block_threads.x] = accum;
            }
            g.sync();

            const int threads_to_the_right = static_cast<int>(x + g.dim_threads().x) - readDF.activeThreads.x;
            const uint activeBlockThreads_x = threads_to_the_right <= 0 ? B_x : B_x - threads_to_the_right;
            const int threads_down = static_cast<int>(y + g.dim_threads().y) - readDF.activeThreads.y;
            const uint activeBlockThreads_y = threads_down <= 0 ? B_y : B_y - threads_down;
            const auto grid = cg::this_grid();
            // Only thread 0 from each TB accumulates from shared memory -> secondDF
            if (g.thread_rank() == 0) {
                AccumType accum{};
                for (int x_it = 0; x_it < activeBlockThreads_x; x_it++) {
                    for (int y_it = 0; y_it < activeBlockThreads_y; y_it++) {
                        const AccumType from_shared = shared_data[y_it][x_it];
                        accum = execute_secondDF(from_shared, accum, reductions...);
                    }
                }

                const uint write_global_position = g.group_index().x + (g.group_index().y * grid.dim_blocks().x);
                WriteIntermediateDF::exec(Point(write_global_position, 0, z), accum, wIDF.params);
            }

            grid.sync();

            // Starting here, we treat the problem as a one dimensional problem
            uint numResults = grid.num_blocks();
            while (numResults >= 1) {
                constexpr uint block_num_threads = B_x * B_y;
                const uint global_thread = g.thread_rank() + (block_num_threads * grid.block_rank());
                const uint block_thread = g.thread_rank();
                AccumType* shared = shared_data;
                if (global_thread < activeThreadBlocks) {
                    // Optimizable with warp reduction
                    const AccumType accum = ReadIntermediateDF::exec(Point(global_thread, 0, z), rIDF.params);
                    shared[block_thread] = accum;
                }
                g.sync();
                if (global_thread < activeThreadBlocks && g.thread_rank() == 0) {
                    const int threads_to_the_right = static_cast<int>(global_thread + g.num_threads()) - activeThreadBlocks;
                    const uint activeBlockThreads = threads_to_the_right <= 0 ? B_x : B_x - threads_to_the_right;
                    AccumType accum{};
                    for (int x_it = 0; x_it < activeBlockThreads; x_it++) {
                        accum = execute_secondDF(shared[block_thread], accum, reductions...);
                    }
                    const uint write_global_position = grid.block_rank();
                    if (numResults > 1) {
                        WriteIntermediateDF::exec(Point(write_global_position, 0, z), accum, wIDF.params);
                    } else {
                        WriteDF::exec(Point(0, 0, z), compute(accum, finalDF), writeDF.params);
                    }
                }
                numResults = numResults == 1 ? 0 : static_cast<uint>(__fdiv_ru(static_cast<float>(numResults), static_cast<float>(block_num_threads)));
                grid.sync();
            }

            if constexpr (ReturnResult) {
                if (x < readDF.activeThreads.x && y < readDF.activeThreads.y) {
                    // All active threads read the result, using L2 and L1 efficiently
                    return { read_value, read(Point(0,0,z), readFDF) };
                } else {
                    return { make_set<ReadType>(0), make_set<OutputType>(0) };
                }
            } else {
                return { make_set<ReadType>(0), make_set<OutputType>(0) };
            }
        }
    };

    template <typename... DataParallelPatterns>
    __global__ void executeDPPs(const DataParallelPatterns&... dpps) {

    }

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformDPP<THREAD_DIVISIBLE, THREAD_FUSION>::exec(deviceFunctionInstances...);
    }

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... DeviceFunctionTypes>
    __global__ void cuda_transform_sequence(const DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformDPP<THREAD_DIVISIBLE, THREAD_FUSION>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformDPP<true, false>::exec(deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform_sequence(const DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformDPP<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ DeviceFunctionTypes... deviceFunctionInstances) {
        TransformDPP<true, false>::exec(deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformDPP<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <typename SequenceSelector, typename... DeviceFunctionSequenceTypes>
    __global__ void cuda_transform_divergent_batch(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformDPP<SequenceSelector>::exec(dfSequenceInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void  __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
        cuda_transform_bounds(const DeviceFunctionTypes... deviceFunctionInstances) {
        TransformDPP<true, false>::exec(deviceFunctionInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
        cuda_transform_bounds(const DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformDPP<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
        cuda_transform_grid_const_bounds(const __grid_constant__ DeviceFunctionTypes... deviceFunctionInstances) {
        TransformDPP<true, false>::exec(deviceFunctionInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... DeviceFunctionTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
        cuda_transform_grid_const_bounds(const __grid_constant__ DeviceFunctionSequence<DeviceFunctionTypes...> deviceFunctionInstances) {
        fk::apply(TransformDPP<true, false>::template exec<DeviceFunctionTypes...>,
            deviceFunctionInstances.deviceFunctions);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename SequenceSelector, typename... DeviceFunctionSequenceTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
        cuda_transform_divergent_batch_bounds(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformDPP<SequenceSelector>::exec(dfSequenceInstances...);
    }
} // namespace fk