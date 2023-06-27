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

#include "operations.cuh"
#include "memory_operations.cuh"
#include "parameter_pack_utils.cuh"

namespace fk { // namespace FusedKernel
// generic operation structs
template <typename Operation_t>
struct ReadDeviceFunction {
    typename Operation_t::ParamsType params;
    dim3 activeThreads;
    using Operation = Operation_t;
};

template <typename Operation_t>
struct BinaryDeviceFunction {
    typename Operation_t::ParamsType params;
    using Operation = Operation_t;
};

template <typename Operation_t>
struct UnaryDeviceFunction {
    using Operation = Operation_t;
};

template <typename Operation_t>
struct MidWriteDeviceFunction {
    typename Operation_t::ParamsType params;
    using Operation = Operation_t;
};

template <typename Operation_t>
struct WriteDeviceFunction {
    typename Operation_t::ParamsType params;
    using Operation = Operation_t;
};

// Recursive operate function
template <typename Operation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data,
                                                  const BinaryDeviceFunction<Operation>& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
    return operate(thread, Operation::exec(i_data, df.params), deviceFunctionInstances...);
}

template <typename Operation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data,
                                                  const UnaryDeviceFunction<Operation>& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
    return operate(thread, Operation::exec(i_data), deviceFunctionInstances...);
}

template <typename Operation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::Type& i_data,
                                                  const MidWriteDeviceFunction<Operation>& df, const DeviceFunctionTypes&... deviceFunctionInstances) {
    Operation::exec(thread, i_data, df.params);
    return operate(thread, i_data, deviceFunctionInstances...);
}

template <typename Operation>
__device__ __forceinline__ constexpr typename Operation::Type operate(const Point& thread, const typename Operation::Type& i_data,
                                                                      const WriteDeviceFunction<Operation>& df) {
    return i_data;
}

template <typename ReadOperation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr void cuda_transform_d(const ReadDeviceFunction<ReadOperation>& readDeviceFunction,
                                                           const DeviceFunctionTypes&... deviceFunctionInstances) {
    const auto writeDeviceFunction = last(deviceFunctionInstances...);
    using WriteOperation = typename decltype(writeDeviceFunction)::Operation;

    const cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
    const Point thread{x, y, z};

    if (x < readDeviceFunction.activeThreads.x && y < readDeviceFunction.activeThreads.y && z < readDeviceFunction.activeThreads.z) {
        const auto tempI = ReadOperation::exec(thread, readDeviceFunction.params);
        if constexpr (sizeof...(deviceFunctionInstances) > 1) {
            const auto tempO = operate(thread, tempI, deviceFunctionInstances...);
            WriteOperation::exec(thread, tempO, writeDeviceFunction.params);
        } else {
            WriteOperation::exec(thread, tempI, writeDeviceFunction.params);
        }
    }
}

template <typename... DeviceFunctionTypes>
__global__ void cuda_transform(const DeviceFunctionTypes... deviceFunctionInstances) {
    cuda_transform_d(deviceFunctionInstances...);
}

template <typename SequenceSelector, int BATCH, int OpSequenceNumber, typename ReadOperation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z,
                                                            const DeviceFunctionSequence<ReadDeviceFunction<ReadOperation>, DeviceFunctionTypes...>& dfSeq) {
    // If the threads with this z, arrived here, we assume they have to execute this operation sequence
    fk::apply(cuda_transform_d<ReadOperation, DeviceFunctionTypes...>, dfSeq.deviceFunctions);
}

template <typename SequenceSelector, int BATCH, int OpSequenceNumber, typename ReadOperation, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z,
                                                            const DeviceFunctionSequence<ReadDeviceFunction<ReadOperation>, DeviceFunctionTypes...>& dfSeq,
                                                            const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
    if (OpSequenceNumber == SequenceSelector::at(z)) {
        fk::apply(cuda_transform_d<ReadOperation, DeviceFunctionTypes...>, dfSeq.deviceFunctions);
    } else {
        divergent_operate<SequenceSelector, BATCH, OpSequenceNumber + 1>(z, dfSequenceInstances...);
    }
}

template <typename SequenceSelector, int BATCH, typename... DeviceFunctionSequenceTypes>
__global__ void cuda_transform_divergent_batch(const DeviceFunctionSequenceTypes... dfSequenceInstances) {
    const cg::thread_block g = cg::this_thread_block();
    const uint z = g.group_index().z;
    divergent_operate<SequenceSelector, BATCH, 1>(z, dfSequenceInstances...);
}

/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
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

template <int BATCH, int OpSequenceNumber, typename ReadOperation, typename... DeviceFunctionTypes>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const Array<int, BATCH>& dfSeqSelector,
                                                            const DeviceFunctionSequence<ReadDeviceFunction<ReadOperation>, DeviceFunctionTypes...>& dfSeq) {
    // If the threads with this z, arrived here, we assume they have to execute this operation sequence
    fk::apply(cuda_transform_d<ReadOperation, DeviceFunctionTypes...>, dfSeq.deviceFunctions);
}

template <int BATCH, int OpSequenceNumber, typename ReadOperation, typename... DeviceFunctionTypes, typename... DeviceFunctionSequenceTypes>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const Array<int, BATCH>& dfSeqSelector,
                                                            const DeviceFunctionSequence<ReadDeviceFunction<ReadOperation>, DeviceFunctionTypes...>& dfSeq,
                                                            const DeviceFunctionSequenceTypes&... dfSequenceInstances) {
    if (OpSequenceNumber == dfSeqSelector.at[z]) {
        fk::apply(cuda_transform_d<ReadOperation, DeviceFunctionTypes...>, dfSeq.deviceFunctions);
    } else {
        divergent_operate<BATCH, OpSequenceNumber + 1>(z, dfSeqSelector, dfSequenceInstances...);
    }
}

template <int BATCH, typename... DeviceFunctionSequenceTypes>
__global__ void cuda_transform_divergent_batch(const Array<int, BATCH> dfSeqSelector, const DeviceFunctionSequenceTypes... dfSequenceInstances) {
    const cg::thread_block g = cg::this_thread_block();
    const uint z = g.group_index().z;
    divergent_operate<BATCH, 1>(z, dfSeqSelector, dfSequenceInstances...);
}
} // namespace FusedKernel
