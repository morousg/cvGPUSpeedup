/*
   Copyright 2023-2024 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>

namespace fk {
    // BatchRead DeviceFunction builders
    template <typename Operation, size_t BATCH, int... Idx>
    constexpr inline std::enable_if_t<isReadBackType<Operation>, SourceReadBack<BatchRead<Operation, BATCH>>> buildBatchReadDF_helper(
        const std::array<typename Operation::ParamsType, BATCH>& params,
        const std::array<typename Operation::BackFunction, BATCH>& back_functions,
        const Size& output_planes,
        const std::integer_sequence<int, Idx...>&) {

        return { {params[Idx]...}, {back_functions[Idx]...},
            {static_cast<uint>(output_planes.width), static_cast<uint>(output_planes.height), static_cast<uint>(BATCH)} };
    }

    template <typename Operation, size_t BATCH, int... Idx>
    constexpr inline std::enable_if_t<isReadType<Operation>, SourceRead<BatchRead<Operation, BATCH>>> buildBatchReadDF_helper(
        const std::array<typename Operation::ParamsType, BATCH>& params,
        const Size& output_planes,
        const std::integer_sequence<int, Idx...>&) {

        return { {params[Idx]...},
            {static_cast<uint>(output_planes.width), static_cast<uint>(output_planes.height), static_cast<uint>(BATCH)} };
    }

    template <typename Operation, size_t BATCH>
    constexpr inline std::enable_if_t<isReadBackType<Operation>, SourceReadBack<BatchRead<Operation, BATCH>>>
        buildBatchReadDF(const std::array<typename Operation::ParamsType, BATCH>& params,
            const std::array<typename Operation::BackFunction, BATCH>& back_functions,
            const Size& output_planes) {

        return buildBatchReadDF_helper<Operation, BATCH>(params, back_functions, output_planes,
            std::make_integer_sequence<int, BATCH>{});
    }

    template <typename Operation, size_t BATCH>
    constexpr inline std::enable_if_t<isReadType<Operation>, SourceRead<BatchRead<Operation, BATCH>>>
        buildBatchReadDF(const std::array<typename Operation::ParamsType, BATCH>& params,
            const Size& output_planes) {

        return buildBatchReadDF_helper<Operation, BATCH>(params, output_planes,
            std::make_integer_sequence<int, BATCH>{});
    }
} // namespace fk
