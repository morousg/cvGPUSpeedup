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

    template <typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const auto readDeviceFunction = first(deviceFunctions...);
        const dim3 dataDims = { readDeviceFunction.activeThreads };
        const dim3 block{ fk::getBlockSize(dataDims.x, dataDims.y) };
        const dim3 grid{ (unsigned int)ceil(dataDims.x / (float)block.x),
                         (unsigned int)ceil(dataDims.y / (float)block.y),
                         dataDims.z };

        cuda_transform << <grid, block, 0, stream >> > (deviceFunctions...);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const dim3 block = input.getBlockSize();
        const dim3 grid{ (uint)ceil(input.dims().width / (float)block.x),
                         (uint)ceil(input.dims().height / (float)block.y) };
        const dim3 gridActiveThreads(input.dims().width, input.dims().height);

        cuda_transform << <grid, block, 0, stream >> > (ReadDeviceFunction<PerThreadRead<_2D, I>>{input, gridActiveThreads}, deviceFunctions...);

        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const dim3 block = input.getBlockSize();
        const dim3 grid((uint)ceil(input.dims().width / (float)block.x), (uint)ceil(input.dims().height / (float)block.y));
        const dim3 gridActiveThreads(input.dims().width, input.dims().height);

        const ReadDeviceFunction<PerThreadRead<_2D, I>> firstOp{ input, gridActiveThreads };
        const WriteDeviceFunction<PerThreadWrite<_2D, O>> opFinal{ output };

        cuda_transform << <grid, block, 0, stream >> > (firstOp, deviceFunctions..., opFinal);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, int Batch, typename... DeviceFunctionTypes>
    inline constexpr void executeOperations(const std::array<fk::Ptr2D<I>, Batch>& input, const int& activeBatch, const cudaStream_t& stream, const DeviceFunctionTypes&... deviceFunctions) {
        const Ptr2D<I>& firstInput = input[0];
        const dim3 block = firstInput.getBlockSize();
        const dim3 grid{ (uint)ceil(firstInput.dims().width / (float)block.x),
                         (uint)ceil(firstInput.dims().height / (float)block.y),
                         (uint)activeBatch };
        const dim3 gridActiveThreads(firstInput.dims().width, firstInput.dims().height, activeBatch);

        Read<BatchRead<PerThreadRead<_2D, I>, Batch>> firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.head.params[plane] = input[plane];
        }
        firstOp.activeThreads = gridActiveThreads;

        cuda_transform << <grid, block, 0, stream >> > (firstOp, deviceFunctions...);
        gpuErrchk(cudaGetLastError());
    }

    template <typename I, typename O, int Batch, typename... operations>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const Tensor<O>& output, const cudaStream_t& stream, const operations&... ops) {
        const Ptr2D<I>& firstInput = input[0];
        const dim3 block = output.getBlockSize();
        const dim3 grid(ceil(firstInput.dims().width / (float)block.x), ceil(firstInput.dims().rows / (float)block.y), activeBatch);
        const dim3 gridActiveThreads(firstInput.dims().width, firstInput.dims().height, activeBatch);

        ReadDeviceFunction<BatchRead<PerThreadRead<_2D, I>, Batch>> firstOp;
        for (int plane = 0; plane < activeBatch; plane++) {
            firstOp.params[plane] = input[plane];
        }
        firstOp.activeThreads = gridActiveThreads;
        const WriteDeviceFunction<PerThreadWrite<_3D, O>> opFinal{ output };

        cuda_transform << <grid, block, 0, stream >> > (firstOp, ops..., opFinal);
        gpuErrchk(cudaGetLastError());
    }
};
