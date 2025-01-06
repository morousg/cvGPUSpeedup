/* Copyright 2023-2025 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/data_parallel_patterns.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/set.cuh>

namespace fk {
    namespace execute_operations_internal {
        template <bool THREAD_FUSION, typename... IOps>
        inline constexpr void executeOperations_helper(const cudaStream_t& stream, const IOps&... iOps) {
            const auto tDetails = TransformDPP<void>::build_details<THREAD_FUSION>(iOps...);
            if constexpr (decltype(tDetails)::TFI::ENABLED) {
                const ActiveThreads activeThreads = tDetails.activeThreads;

                const dim3 block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                if (!tDetails.threadDivisible) {
                    launchTransformDPP_Kernel<false><<<grid, block, 0, stream>>>(tDetails, iOps...);
                } else {
                    launchTransformDPP_Kernel<true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                }
            } else {
                const auto readOp = get<0>(iOps...);

                const ActiveThreads activeThreads = readOp.getActiveThreads();

                const dim3 block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };

                launchTransformDPP_Kernel<true><<<grid, block, 0, stream>>>(tDetails, iOps...);
            }
            gpuErrchk(cudaGetLastError());
        }
    } // namespace execute_operations_internal

    template <bool THREAD_FUSION, typename... IOps>
    inline constexpr void executeOperations(const cudaStream_t& stream,
                                            const IOps&... iOps) {
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream, iOps...);
    }

    template <typename... IOps>
    inline constexpr void executeOperations(const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(stream, iOps...);
    }

    template <bool THREAD_FUSION, typename I, typename... IOps>
    inline constexpr void executeOperations(const Ptr2D<I>& input,
                                            const cudaStream_t& stream,
                                            const IOps&... iOps) {
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream,
            PerThreadRead<_2D, I>::build({ input }), iOps...);
    }

    template <typename I, typename... IOps>
    inline constexpr void executeOperations(const Ptr2D<I>& input,
                                            const cudaStream_t& stream,
                                            const IOps&... iOps) {
        executeOperations<true>(input, stream, iOps...);
    }

    template <bool THREAD_FUSION, typename I, typename O, typename... IOps>
    inline constexpr void executeOperations(const Ptr2D<I>& input,
                                            const Ptr2D<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream,
            PerThreadRead<_2D, I>::build({ input }), iOps..., PerThreadWrite<_2D, O>::build({ output }));
    }

    template <typename I, typename O, typename... IOps>
    inline constexpr void executeOperations(const Ptr2D<I>& input,
                                            const Ptr2D<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(input, output, stream, iOps...);
    }

    template <bool THREAD_FUSION, typename I, int BATCH, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                            const int& activeBatch, const I& defaultValue,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream, batchReadIOp, iOps...);
    }

    template <bool THREAD_FUSION, typename I, int BATCH, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream, batchReadIOp, iOps...);
    }

    template <typename I, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const int& activeBatch, const I& defaultValue,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(input, activeBatch, defaultValue, stream, iOps...);
    }

    template <typename I, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(input, stream, iOps...);
    }

    template <bool THREAD_FUSION, typename I, typename O, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const int& activeBatch, const I& defaultValue,
                                            const Tensor<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
        const auto writeOp = PerThreadWrite<_3D, O>::build(output);
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream, batchReadIOp, iOps..., writeOp);
    }

    template <bool THREAD_FUSION, typename I, typename O, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const Tensor<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
        const auto writeOp = PerThreadWrite<_3D, O>::build(output);
        execute_operations_internal::executeOperations_helper<THREAD_FUSION>(stream, batchReadIOp, iOps..., writeOp);
    }

    template <typename I, typename O, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const Tensor<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(input, output, stream, iOps...);
    }

    template <typename I, typename O, int Batch, typename... IOps>
    inline constexpr void executeOperations(const std::array<Ptr2D<I>, Batch>& input,
                                            const int& activeBatch, const I& defaultValue,
                                            const Tensor<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
        executeOperations<true>(input, activeBatch, defaultValue, output, stream, iOps...);
    }

    template <ND D, typename T>
    inline constexpr void setTo(const T& value, Ptr<D, T>& outputPtr, const cudaStream_t& stream = 0) {
        RawPtr<D, T> output = outputPtr.ptr();
        if (outputPtr.getMemType() == MemType::Device) {
            if constexpr (D == _1D) {
                const ActiveThreads activeThreads(output.dims.width);
                executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build(output));
            } else if constexpr (D == _2D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height);
                executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build(output));
            } else if constexpr (D == _3D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height, output.dims.planes);
                executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build(output));
            }
        } else {
            for (int i = 0; i < (int)outputPtr.getNumElements(); i++) {
                output.data[i] = value;
            }
        }
    }
} // namespace fk

#endif
