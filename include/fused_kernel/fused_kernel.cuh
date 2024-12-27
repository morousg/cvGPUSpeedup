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

#ifndef FK_FUSED_KERNEL
#define FK_FUSED_KERNEL

#include <fused_kernel/core/execution_model/grid_patterns.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/set.cuh>

namespace fk {
    template <typename ReadInstantiableOperation, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadInstantiableOperation& readDF, const InstantiableOperationTypes&... instantiableOperations) {
        executeOperations<true>(stream, readDF, instantiableOperations...);
    }

    template <bool THREAD_FUSION, typename ReadInstantiableOperation, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadInstantiableOperation& readDF, const InstantiableOperationTypes&... instantiableOperations) {
        const ActiveThreads dataDims = readDF.activeThreads;
        const dim3 block{ getDefaultBlockSize(dataDims.x, dataDims.y) };
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadInstantiableOperation, InstantiableOperationTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(readDF, instantiableOperations...);
        const dim3 grid{ (unsigned int)ceil((dataDims.x/ (float)elems_per_thread) / (float)block.x),
                         (unsigned int)ceil(dataDims.y / (float)block.y),
                         dataDims.z };

        const ActiveThreads activeThreads{ (uint)ceil(readDF.activeThreads.x / (float)elems_per_thread), readDF.activeThreads.y, readDF.activeThreads.z };

        ReadInstantiableOperation readInstantiableOperation = readDF;
        readInstantiableOperation.activeThreads = activeThreads;

        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, readInstantiableOperation, instantiableOperations...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
        }

        gpuErrchk(cudaGetLastError());
    }

    template <bool THREAD_FUSION, typename I, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        using ReadInstantiableOperation = SourceRead<PerThreadRead<_2D, I>>;
        ReadInstantiableOperation readInstantiableOperation{ {input} };
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadInstantiableOperation, InstantiableOperationTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(readInstantiableOperation, instantiableOperations...);

        const dim3 block = input.getBlockSize();
        const dim3 grid{ (uint)ceil(input.dims().width / ((float)elems_per_thread * (float)block.x)),
                         (uint)ceil(input.dims().height / (float)block.y) };
        const ActiveThreads gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        readInstantiableOperation.activeThreads = gridActiveThreads;

        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, readInstantiableOperation, instantiableOperations...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (readInstantiableOperation, instantiableOperations...);
        }

        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        executeOperations<true>(input, stream, instantiableOperations...);
    }

    template <bool THREAD_FUSION, typename I, typename O, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        auto firstOp = PerThreadRead<_2D, I>::build_source(input);
        const auto opFinal = PerThreadWrite<_2D, O>::build(output);
        
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, decltype(firstOp), InstantiableOperationTypes..., decltype(opFinal)>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, instantiableOperations..., opFinal);

        const dim3 block = input.getBlockSize();
        const dim3 grid((uint)ceil(input.dims().width / (elems_per_thread * (float)block.x)), (uint)ceil(input.dims().height / (float)block.y));
        const ActiveThreads gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, instantiableOperations..., opFinal);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations..., opFinal);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations..., opFinal);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, instantiableOperations..., opFinal);
        }
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        executeOperations<true>(input, output, stream, instantiableOperations...);
    }

    template <bool THREAD_FUSION, typename I, int BATCH, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        const Ptr2D<I>& firstInput = input[0];
        using ReadInstantiableOperation = SourceRead<BatchRead<BATCH, PerThreadRead<_2D, I>>>;
        ReadInstantiableOperation firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params.op_params[plane] = input[plane];
        }

        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadInstantiableOperation, InstantiableOperationTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, instantiableOperations...);

        const dim3 block = firstInput.getBlockSize();
        const dim3 grid{ (uint)ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)),
                         (uint)ceil(firstInput.dims().height / (float)block.y),
                         (uint)activeBatch };
        const ActiveThreads gridActiveThreads((uint)ceil(firstInput.dims().width / (float)elems_per_thread), firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, instantiableOperations...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, instantiableOperations...);
        }
        
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, int Batch, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        executeOperations<true>(input, activeBatch, stream, instantiableOperations...);
    }

    template <bool THREAD_FUSION, typename I, typename O, int Batch, typename... InstantiableOperationTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const InstantiableOperationTypes&... instantiableOperations) {
        const Ptr2D<I>& firstInput = input[0];
        
        using ReadDF = SourceRead<BatchRead<Batch, PerThreadRead<_2D, I>>>;
        ReadDF firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }

        using WriteInstantiableOperation = WriteInstantiableOperation<PerThreadWrite<_3D, O>>;
        const WriteInstantiableOperation opFinal{ output };

        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadInstantiableOperation, InstantiableOperationTypes..., WriteInstantiableOperation>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, instantiableOperations..., opFinal);

        const dim3 block = output.getBlockSize();
        const dim3 grid(ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)), ceil(firstInput.dims().rows / (float)block.y), activeBatch);
        const dim3 gridActiveThreads(firstInput.dims().width / (float)elems_per_thread, firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, instantiableOperations..., opFinal);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations..., opFinal);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, instantiableOperations..., opFinal);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, instantiableOperations...);
        }
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, int Batch, typename... operations>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const operations&... ops) {
        executeOperations<true>(input, activeBatch, output, stream, ops...);
    }

    template <ND D, typename T>
    inline constexpr void setTo(const T& value, Ptr<D, T>& outputPtr, const cudaStream_t& stream = 0) {
        RawPtr<D, T> output = outputPtr.ptr();
        if (outputPtr.getMemType() == MemType::Device) {
            if constexpr (D == _1D) {
                const ActiveThreads activeThreads(output.dims.width);
                executeOperations(stream, SourceRead<ReadSet<T>>{value, activeThreads}, Write<PerThreadWrite<D, T>>{output});
            } else if constexpr (D == _2D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height);
                executeOperations(stream, SourceRead<ReadSet<T>>{value, activeThreads}, Write<PerThreadWrite<D, T>>{output});
            } else if constexpr (D == _3D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height, output.dims.planes);
                executeOperations(stream, SourceRead<ReadSet<T>>{value, activeThreads}, Write<PerThreadWrite<D, T>>{output});
            }
        } else {
            for (int i = 0; i < (int)outputPtr.getNumElements(); i++) {
                output.data[i] = value;
            }
        }
    }
} // namespace fk

#endif
