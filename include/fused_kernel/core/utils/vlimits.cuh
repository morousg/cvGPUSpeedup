/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include <fused_kernel/core/utils/cuda_vector_utils.cuh>
#include <limits>

namespace fk{
    // Limits
    template <typename T, typename Enabler = void>
    constexpr T maxValue{};
    template <typename T>
    constexpr T maxValue <T, std::enable_if_t<!validCUDAVec<T> && !std::is_aggregate_v<T>>> = std::numeric_limits<T>::max();
    template <typename T>
    constexpr T maxValue <T, std::enable_if_t<validCUDAVec<T>>> = make_set<T>(maxValue<VBase<T>>);

    template <typename T, typename Enabler = void>
    constexpr T minValue{};
    template <typename T>
    constexpr T minValue <T, std::enable_if_t<!validCUDAVec<T> && !std::is_aggregate_v<T>>> = std::numeric_limits<T>::min();
    template <typename T>
    constexpr T minValue <T, std::enable_if_t<validCUDAVec<T>>> = make_set<T>(minValue<VBase<T>>);
}