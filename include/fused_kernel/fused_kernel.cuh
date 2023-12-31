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

    template <bool THREAD_FUSION, typename... DeviceFuntionTypes>
    inline constexpr uint computeElementsPerThread() {
        if constexpr (THREAD_FUSION) {
            using ReadOperation = typename FirstType_t<DeviceFuntionTypes...>::Operation;
            using WriteOperation = typename LastType_t<DeviceFuntionTypes...>::Operation;
            using RDT = typename ReadOperation::ReadDataType;
            using WDT = typename WriteOperation::WriteDataType;
            
            constexpr bool TF_ENABLED = ReadOperation::THREAD_FUSION && WriteOperation::THREAD_FUSION && THREAD_FUSION;
            return ThreadFusionInfo<RDT, WDT, TF_ENABLED>::elems_per_thread;
        } else {
            return 1u;
        }
    }

    template <bool THREAD_FUSION, typename... DeviceFunctionTypes>
    inline constexpr void executeOperationsImplementation(const cudaStream_t& stream, const dim3& grid, const dim3& block, const uint dataDimsX,
                                                          const uint& elems_per_thread, const DeviceFunctionTypes&... deviceFunctions) {
        const uint thread_fusion_residue = dataDimsX % elems_per_thread;
        if (thread_fusion_residue == 0) {
            cuda_transform<true, THREAD_FUSION> << <grid, block, 0, stream >> > (deviceFunctions...);
        } else {
            cuda_transform<false, THREAD_FUSION> << <grid, block, 0, stream >> > (deviceFunctions...);
        }
    }

    template <typename ReadDeviceFunction, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadDeviceFunction& readDF, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(stream, readDF, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename ReadDeviceFunction, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const ReadDeviceFunction& readDF, const DeviceFunctionTypes&... deviceFunctions) {
        const dim3 dataDims = { readDF.activeThreads };
        const dim3 block{ fk::getBlockSize(dataDims.x, dataDims.y) };
        constexpr uint elems_per_thread = computeElementsPerThread<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes...>();
        const dim3 grid{ (unsigned int)ceil((dataDims.x/ (float)elems_per_thread) / (float)block.x),
                         (unsigned int)ceil(dataDims.y / (float)block.y),
                         dataDims.z };

        const dim3 activeThreads{ (uint)ceil(readDF.activeThreads.x / (float)elems_per_thread), readDF.activeThreads.y, readDF.activeThreads.z };

        ReadDeviceFunction readDeviceFunction;
        using ParsType = typename ReadDeviceFunction::Operation::ParamsType;
        if constexpr (std::is_array_v<ParsType>) {
            std::copy(std::begin(readDF.params), std::end(readDF.params), std::begin(readDeviceFunction.params));
        } else {
            readDeviceFunction.params = readDF.params;
        }
        readDeviceFunction.activeThreads = activeThreads;

        executeOperationsImplementation<THREAD_FUSION>(stream, grid, block, dataDims.x, elems_per_thread, readDeviceFunction, deviceFunctions...);

        gpuErrchk(cudaGetLastError());
    }

    template <bool THREAD_FUSION, typename I, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        using ReadDeviceFunction = Read<PerThreadRead<_2D, I>>;

        constexpr uint elems_per_thread = computeElementsPerThread<THREAD_FUSION, ReadDeviceFunction, DeviceFunctionTypes...>();

        const dim3 block = input.getBlockSize();
        const dim3 grid{ (uint)ceil(input.dims().width / ((float)elems_per_thread * (float)block.x)),
                         (uint)ceil(input.dims().height / (float)block.y) };
        const dim3 gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        const ReadDeviceFunction readDeviceFunction{ input, gridActiveThreads };
        executeOperationsImplementation<THREAD_FUSION>(stream, grid, block, input.dims().width, elems_per_thread, readDeviceFunction, deviceFunctions...);

        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, typename O, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        constexpr uint elems_per_thread = computeElementsPerThread<THREAD_FUSION, ReadDeviceFunction<PerThreadRead<_2D, I>>, DeviceFunctionTypes..., WriteDeviceFunction<PerThreadWrite<_2D, O>>>();

        const dim3 block = input.getBlockSize();
        const dim3 grid((uint)ceil(input.dims().width / (elems_per_thread * (float)block.x)), (uint)ceil(input.dims().height / (float)block.y));
        const dim3 gridActiveThreads((uint)ceil(input.dims().width / (float)elems_per_thread), input.dims().height);
        const ReadDeviceFunction<PerThreadRead<_2D, I>> firstOp{ input, gridActiveThreads };
        const WriteDeviceFunction<PerThreadWrite<_2D, O>> opFinal{ output };
        executeOperationsImplementation<THREAD_FUSION>(stream, grid, block, input.dims().width, elems_per_thread, firstOp, deviceFunctions..., opFinal);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, output, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<fk::Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const Ptr2D<I>& firstInput = input[0];

        Read<BatchRead<PerThreadRead<_2D, I>, Batch>> firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }

        constexpr uint elems_per_thread = computeElementsPerThread<THREAD_FUSION, Read<BatchRead<PerThreadRead<_2D, I>, Batch>>, DeviceFunctionTypes...>();
        const dim3 block = firstInput.getBlockSize();
        const dim3 grid{ (uint)ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)),
                         (uint)ceil(firstInput.dims().height / (float)block.y),
                         (uint)activeBatch };
        const dim3 gridActiveThreads((uint)ceil(firstInput.dims().width / (float)elems_per_thread), firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        executeOperationsImplementation<THREAD_FUSION>(stream, grid, block, firstInput.dims().width, elems_per_thread, firstOp, deviceFunctions...);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<fk::Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        executeOperations<true>(input, activeBatch, stream, deviceFunctions...);
    }

    template <bool THREAD_FUSION, typename I, typename O, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... ops) {
        const Ptr2D<I>& firstInput = input[0];
        
        ReadDeviceFunction<BatchRead<PerThreadRead<_2D, I>, Batch>> firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }

        const WriteDeviceFunction<PerThreadWrite<_3D, O>> opFinal{ output };

        constexpr uint elems_per_thread = computeElementsPerThread<THREAD_FUSION, ReadDeviceFunction<BatchRead<PerThreadRead<_2D, I>, Batch>>, DeviceFunctionTypes..., WriteDeviceFunction<PerThreadWrite<_3D, O>>>();

        const dim3 block = output.getBlockSize();
        const dim3 grid(ceil(firstInput.dims().width / (elems_per_thread * (float)block.x)), ceil(firstInput.dims().rows / (float)block.y), activeBatch);
        const dim3 gridActiveThreads(firstInput.dims().width / (float)elems_per_thread, firstInput.dims().height, activeBatch);
        firstOp.activeThreads = gridActiveThreads;
        executeOperationsImplementation<THREAD_FUSION>(stream, grid, block, firstInput.dims().width, elems_per_thread, firstOp, ops..., opFinal);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, int Batch, typename... operations>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const operations&... ops) {
        executeOperations<true>(input, activeBatch, output, stream, ops...);
    }
};
