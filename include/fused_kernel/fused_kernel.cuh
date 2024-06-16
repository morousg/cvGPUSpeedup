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

#include <fused_kernel/core/execution_model/grid_patterns.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>

namespace fk {
    template <typename ReadDeviceFunction, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadDeviceFunction& readDF, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(stream, readDF, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename ReadDeviceFunction, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadDeviceFunction& readDF, const DeviceFunctionTypes&... deviceFunctions) {
        const dim3 dataDims = { readDF.activeThreads };
        const dim3 block{ getBlockSize(dataDims.x, dataDims.y) };
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(readDF, deviceFunctions...);
        const dim3 grid{ (unsigned int)ceil((dataDims.x/ (float)elems_per_thread) / (float)block.x),
                         (unsigned int)ceil(dataDims.y / (float)block.y),
                         dataDims.z };

        const dim3 activeThreads{ (uint)ceil(readDF.activeThreads.x / (float)elems_per_thread), readDF.activeThreads.y, readDF.activeThreads.z };

        ReadDeviceFunction readDeviceFunction = readDF;
        readDeviceFunction.activeThreads = activeThreads;

        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, readDeviceFunction, deviceFunctions...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
        }

        gpuErrchk(cudaGetLastError());
    }

    template <bool THREAD_FUSION, typename I, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        using ReadDeviceFunction = SourceRead<PerThreadRead<_2D, I>>;
        ReadDeviceFunction readDeviceFunction{ input };
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(readDeviceFunction, deviceFunctions...);

        const dim3 block = input.getBlockSize();
        const dim3 grid{ (uint)ceil(input.dims().width / ((float)elems_per_thread * (float)block.x)),
                         (uint)ceil(input.dims().height / (float)block.y) };
        const dim3 gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        readDeviceFunction.activeThreads = gridActiveThreads;

        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, readDeviceFunction, deviceFunctions...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (readDeviceFunction, deviceFunctions...);
        }

        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, typename O, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        using ReadDeviceFunction = SourceReadDeviceFunction<PerThreadRead<_2D, I>>;
        ReadDeviceFunction firstOp{ input };
        using WriteDeviceFunction = WriteDeviceFunction<PerThreadWrite<_2D, O>>;
        const WriteDeviceFunction opFinal{ output };
        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes..., WriteDeviceFunction>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, deviceFunctions..., opFinal);

        const dim3 block = input.getBlockSize();
        const dim3 grid((uint)ceil(input.dims().width / (elems_per_thread * (float)block.x)), (uint)ceil(input.dims().height / (float)block.y));
        const dim3 gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, deviceFunctions..., opFinal);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
        }
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, output, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const Ptr2D<I>& firstInput = input[0];
        using ReadDeviceFunction = SourceRead<BatchRead<PerThreadRead<_2D, I>, Batch>>;
        ReadDeviceFunction firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }

        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes...>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, deviceFunctions...);

        const dim3 block = firstInput.getBlockSize();
        const dim3 grid{ (uint)ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)),
                         (uint)ceil(firstInput.dims().height / (float)block.y),
                         (uint)activeBatch };
        const dim3 gridActiveThreads((uint)ceil(firstInput.dims().width / (float)elems_per_thread), firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, deviceFunctions...);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions...);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions...);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, deviceFunctions...);
        }
        
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, activeBatch, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, typename O, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const Ptr2D<I>& firstInput = input[0];
        
        using ReadDF = SourceRead<BatchRead<PerThreadRead<_2D, I>, Batch>>;
        ReadDF firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }

        using WriteDeviceFunction = WriteDeviceFunction<PerThreadWrite<_3D, O>>;
        const WriteDeviceFunction opFinal{ output };

        constexpr bool THREAD_FUSION_ENABLED = isThreadFusionEnabled<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes..., WriteDeviceFunction>();
        const uint elems_per_thread = computeElementsPerThread<THREAD_FUSION_ENABLED>(firstOp, deviceFunctions..., opFinal);

        const dim3 block = output.getBlockSize();
        const dim3 grid(ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)), ceil(firstInput.dims().rows / (float)block.y), activeBatch);
        const dim3 gridActiveThreads(firstInput.dims().width / (float)elems_per_thread, firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        if (elems_per_thread > 1) {
            const bool threadDisvisible = isThreadDivisible<THREAD_FUSION_ENABLED>(elems_per_thread, firstOp, deviceFunctions..., opFinal);
            if (threadDisvisible) {
                cuda_transform<true, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
            } else {
                cuda_transform<false, THREAD_FUSION_ENABLED> << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
            }
        } else {
            cuda_transform<true, false> << <grid, block, 0, stream >> > (firstOp, deviceFunctions...);
        }
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, int Batch, typename... operations>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const operations&... ops) {
        executeOperations<true>(input, activeBatch, output, stream, ops...);
    }
} // namespace fk
