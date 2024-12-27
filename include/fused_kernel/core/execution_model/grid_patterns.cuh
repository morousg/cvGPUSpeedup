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

#ifndef FK_GRID_PATTERNS
#define FK_GRID_PATTERNS

#include <cooperative_groups.h>

namespace cooperative_groups {};
namespace cg = cooperative_groups;

#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION>
    struct TransformGridPattern {
        private:
            template <typename T, typename InstantiableOperation, typename... InstantiableOperationTypes>
            FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const InstantiableOperation& df, const InstantiableOperationTypes&... instantiableOperationInstances) {
                if constexpr (InstantiableOperation::template is<WriteType>) {
                    return i_data;
                // MidWriteOperation with continuations, based on FusedOperation
                } else if constexpr (InstantiableOperation::template is<MidWriteType> && isMidWriteType<typename InstantiableOperation::Operation>) {
                    return InstantiableOperation::Operation::exec(thread, i_data, df.params);
                } else if constexpr (InstantiableOperation::template is<MidWriteType> && !isMidWriteType<typename InstantiableOperation::Operation>) {
                    InstantiableOperation::Operation::exec(thread, i_data, df.params);
                    return i_data;
                } else {
                    return operate(thread, compute(i_data, df), instantiableOperationInstances...);
                }
            }

            template <uint IDX, typename TFI, typename InputType, typename... InstantiableOperationTypes>
            FK_DEVICE_FUSE auto operate_idx(const Point& thread, const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
                return operate(thread, TFI::get<IDX>(input), instantiableOperationInstances...);
            }

            template <typename TFI, typename InputType, uint... IDX, typename... InstantiableOperationTypes>
            FK_DEVICE_FUSE auto operate_thread_fusion_impl(std::integer_sequence<uint, IDX...> idx, const Point& thread,
                                                           const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
                return TFI::make(operate_idx<IDX, TFI>(thread, input, instantiableOperationInstances...)...);
            }

            template <typename TFI, typename InputType, typename... InstantiableOperationTypes>
            FK_DEVICE_FUSE auto operate_thread_fusion(const Point& thread, const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
                if constexpr (TFI::elems_per_thread == 1) {
                    return operate(thread, input, instantiableOperationInstances...);
                } else {
                    return operate_thread_fusion_impl<TFI>(std::make_integer_sequence<uint, TFI::elems_per_thread>(), thread, input, instantiableOperationInstances...);
                }
            }

            template <typename ReadInstantiableOperation, typename TFI>
            FK_DEVICE_FUSE auto read(const Point& thread, const ReadInstantiableOperation& readDF) {
                if constexpr (ReadInstantiableOperation::template is<ReadBackType>) {
                    if constexpr (TFI::ENABLED) {
                        return ReadInstantiableOperation::Operation::exec<TFI::elems_per_thread>(thread, readDF.params, readDF.back_function);
                    } else {
                        return ReadInstantiableOperation::Operation::exec(thread, readDF.params, readDF.back_function);
                    }
                } else if constexpr (ReadInstantiableOperation::template is<ReadType>) {
                    if constexpr (TFI::ENABLED) {
                        return ReadInstantiableOperation::Operation::exec<TFI::elems_per_thread>(thread, readDF.params);
                    } else {
                        return ReadInstantiableOperation::Operation::exec(thread, readDF.params);
                    }
                }
            }

            template <typename TFI, typename ReadInstantiableOperation, typename... InstantiableOperations>
            FK_DEVICE_FUSE void execute_instantiable_operations(const Point& thread, const ReadInstantiableOperation& readDF,
                                                       const InstantiableOperations&... instantiableOperationInstances) {
                using ReadOperation = typename ReadInstantiableOperation::Operation;
                using WriteOperation = typename LastType_t<InstantiableOperations...>::Operation;

                const auto writeDF = ppLast(instantiableOperationInstances...);

                if constexpr (TFI::ENABLED) {
                    const auto tempI = read<ReadInstantiableOperation, TFI>(thread, readDF);
                    if constexpr (sizeof...(instantiableOperationInstances) > 1) {
                        const auto tempO = operate_thread_fusion<TFI>(thread, tempI, instantiableOperationInstances...);
                        WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDF.params);
                    } else {
                        WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDF.params);
                    }
                } else {
                    const auto tempI = read<ReadInstantiableOperation, TFI>(thread, readDF);
                    if constexpr (sizeof...(instantiableOperationInstances) > 1) {
                        const auto tempO = operate(thread, tempI, instantiableOperationInstances...);
                        WriteOperation::exec(thread, tempO, writeDF.params);
                    } else {
                        WriteOperation::exec(thread, tempI, writeDF.params);
                    }
                }
            }

        public:
            template <typename ReadInstantiableOperation, typename... InstantiableOperationTypes>
            FK_DEVICE_FUSE void exec(const ReadInstantiableOperation& readInstantiableOperation, const InstantiableOperationTypes&... instantiableOperationInstances) {
                const cg::thread_block g = cg::this_thread_block();

                const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
                const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
                const uint z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
                const Point thread{ x, y, z };

                using ReadOperation = typename ReadInstantiableOperation::Operation;
                using WriteOperation = typename LastType_t<InstantiableOperationTypes...>::Operation;
                using ReadIT = typename ReadOperation::ReadDataType;
                using WriteOT = typename WriteOperation::WriteDataType;
                using TFI = ThreadFusionInfo<ReadIT, WriteOT,
                                             and_v<ReadOperation::THREAD_FUSION, WriteOperation::THREAD_FUSION, THREAD_FUSION>>;

                if (x < readInstantiableOperation.activeThreads.x && y < readInstantiableOperation.activeThreads.y) {
                    if constexpr (THREAD_DIVISIBLE || !TFI::ENABLED) {
                        execute_instantiable_operations<TFI>(thread, readInstantiableOperation, instantiableOperationInstances...);
                    } else {
                        if (x < readInstantiableOperation.activeThreads.x - 1) {
                            execute_instantiable_operations<TFI>(thread, readInstantiableOperation, instantiableOperationInstances...);
                        } else if (x == readInstantiableOperation.activeThreads.x - 1) {
                            const uint initialX = x * TFI::elems_per_thread;
                            const uint finalX = ReadOperation::num_elems_x(thread, readInstantiableOperation.params);
                            uint currentX = initialX;
                            while (currentX < finalX) {
                                const Point currentThread{ currentX , thread.y, thread.z };
                                using DisabledTFI = ThreadFusionInfo<ReadIT, WriteOT, false>;
                                execute_instantiable_operations<DisabledTFI>(currentThread, readInstantiableOperation, instantiableOperationInstances...);
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
            template <int OpSequenceNumber, typename... InstantiableOperationTypes, typename... InstantiableOperationSequenceTypes>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const InstantiableOperationSequence<InstantiableOperationTypes...>& dfSeq,
                                                  const InstantiableOperationSequenceTypes&... dfSequenceInstances) {
                if (OpSequenceNumber == SequenceSelector::at(z)) {
                    apply(TransformGridPattern<true, false>::exec<InstantiableOperationTypes...>, dfSeq.instantiableOperations);
                } else if constexpr (sizeof...(dfSequenceInstances) > 0) {
                    divergent_operate<OpSequenceNumber + 1>(z, dfSequenceInstances...);
                }
            }
        public:
            template <typename... InstantiableOperationSequenceTypes>
            FK_DEVICE_FUSE void exec(const InstantiableOperationSequenceTypes&... dfSequenceInstances) {
                const cg::thread_block g = cg::this_thread_block();
                const uint z = g.group_index().z;
                divergent_operate<1>(z, dfSequenceInstances...);
            }
    };

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... InstantiableOperationTypes>
    __global__ void cuda_transform(const InstantiableOperationTypes... instantiableOperationInstances) {
        TransformGridPattern<THREAD_DIVISIBLE, THREAD_FUSION>::exec(instantiableOperationInstances...);
    }

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... InstantiableOperationTypes>
    __global__ void cuda_transform_sequence(const InstantiableOperationSequence<InstantiableOperationTypes...> instantiableOperationInstances) {
        apply(TransformGridPattern<THREAD_DIVISIBLE, THREAD_FUSION>::template exec<InstantiableOperationTypes...>,
            instantiableOperationInstances.instantiableOperations);
    }

    template <typename... InstantiableOperationTypes>
    __global__ void cuda_transform(const InstantiableOperationTypes... instantiableOperationInstances) {
        TransformGridPattern<true, false>::exec(instantiableOperationInstances...);
    }

    template <typename... InstantiableOperationTypes>
    __global__ void cuda_transform_sequence(const InstantiableOperationSequence<InstantiableOperationTypes...> instantiableOperationInstances) {
        apply(TransformGridPattern<true, false>::template exec<InstantiableOperationTypes...>,
            instantiableOperationInstances.instantiableOperations);
    }

    template <typename... InstantiableOperationTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ InstantiableOperationTypes... instantiableOperationInstances) {
        TransformGridPattern<true, false>::exec(instantiableOperationInstances...);
    }

    template <typename... InstantiableOperationTypes>
    __global__ void cuda_transform_grid_const(const __grid_constant__ InstantiableOperationSequence<InstantiableOperationTypes...> instantiableOperationInstances) {
        apply(TransformGridPattern<true, false>::template exec<InstantiableOperationTypes...>,
            instantiableOperationInstances.instantiableOperations);
    }

    template <typename SequenceSelector, typename... InstantiableOperationSequenceTypes>
    __global__ void cuda_transform_divergent_batch(const InstantiableOperationSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern<SequenceSelector>::exec(dfSequenceInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... InstantiableOperationTypes>
    __global__ void  __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_bounds(const InstantiableOperationTypes... instantiableOperationInstances) {
        TransformGridPattern<true, false>::exec(instantiableOperationInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... InstantiableOperationTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_bounds(const InstantiableOperationSequence<InstantiableOperationTypes...> instantiableOperationInstances) {
        apply(TransformGridPattern<true, false>::template exec<InstantiableOperationTypes...>,
            instantiableOperationInstances.instantiableOperations);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... InstantiableOperationTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_grid_const_bounds(const __grid_constant__ InstantiableOperationTypes... instantiableOperationInstances) {
        TransformGridPattern<true, false>::exec(instantiableOperationInstances...);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename... InstantiableOperationTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_grid_const_bounds(const __grid_constant__ InstantiableOperationSequence<InstantiableOperationTypes...> instantiableOperationInstances) {
        apply(TransformGridPattern<true, false>::template exec<InstantiableOperationTypes...>,
            instantiableOperationInstances.instantiableOperations);
    }

    template <int MAX_THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MP, typename SequenceSelector, typename... InstantiableOperationSequenceTypes>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    cuda_transform_divergent_batch_bounds(const InstantiableOperationSequenceTypes... dfSequenceInstances) {
        DivergentBatchTransformGridPattern<SequenceSelector>::exec(dfSequenceInstances...);
    }
} // namespace fk

#endif
