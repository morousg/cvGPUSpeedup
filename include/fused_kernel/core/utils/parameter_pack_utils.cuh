/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
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

#ifndef FK_PARAMETER_PACK_UTILS
#define FK_PARAMETER_PACK_UTILS

#include <fused_kernel/core/utils/utils.h>

namespace fk { // namespace fused kernel

    // Util to get the parameters of a parameter pack
    template <int Index, typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto ppGet(const T& current, const Args&... args) {
        static_assert(sizeof...(args) + 1 > Index, "Index out of range when looking for a parameter in a parameter pack.");
        if constexpr (Index == 0) {
            return current;
        } else {
            return ppGet<Index - 1>(args...);
        }
    }

    // Util to get the last parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST auto ppLast(const Args&... args) {
        return ppGet<sizeof...(args) - 1>(args...);
    }

    // Util to get the first parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST auto ppFirst(const Args&... args) {
        return ppGet<0>(args...);
    }

    template <size_t idx, size_t... iseq>
    constexpr inline size_t get_index_f(const std::index_sequence<iseq...>&) {
        return ppGet<static_cast<int>(idx)>(iseq...);
    }

    template <size_t idx, typename ISeq>
    constexpr size_t get_index = get_index_f<idx>(ISeq{});

    template <typename T, T idx, T... iseq>
    constexpr inline size_t get_integer_f(const std::integer_sequence<T, iseq...>&) {
        return ppGet<static_cast<int>(idx)>(iseq...);
    }

    template <typename T, T idx, typename ISeq>
    constexpr T get_integer = get_integer_f<T, idx>(ISeq{});
} // namespace fk

#endif
