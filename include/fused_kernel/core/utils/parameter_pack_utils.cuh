/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2023 Oscar Amoros Huguet

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

#include <fused_kernel/core/utils/cuda_utils.cuh>
#include <fused_kernel/core/utils/tuple.cuh>

namespace fk { // namespace fused kernel

    template <typename F, typename Tuple, size_t... I>
    FK_HOST_DEVICE_CNST auto apply_impl(F&& f, Tuple& t, std::index_sequence<I...>)
        -> decltype(std::forward<F>(f)(get_v<I>(std::forward<Tuple>(t))...))
    {
        return std::forward<F>(f)(get_v<I>(std::forward<Tuple>(t))...);
    }

    template <typename F, typename Tuple>
    FK_HOST_DEVICE_CNST auto apply(F&& f, Tuple& t)
        -> decltype(apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<Tuple::size>()))
    {
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<Tuple::size>());
    }

    // Struct to hold a parameter pack, and be able to pass it arround
    template <typename... DeviceFunctionTypes>
    struct DeviceFunctionSequence {
        Tuple<const DeviceFunctionTypes...> deviceFunctions;
    };

    // Function that fills the OperationSequence struct, from a parameter pack
    template <typename... DeviceFunctionTypes>
    FK_HOST_DEVICE_CNST auto buildOperationSequence(const DeviceFunctionTypes&... deviceFunctionInstances) {
        return DeviceFunctionSequence<DeviceFunctionTypes...> {{deviceFunctionInstances...}};
    }

    template <typename... DeviceFunctionTypes>
    FK_HOST_DEVICE_CNST auto buildOperationSequence_tup(const Tuple<DeviceFunctionTypes...>& deviceFunctionInstances) {
        return fk::apply([](const auto&... args) { 
                            return buildOperationSequence(args...); 
                         }, deviceFunctionInstances);
    }

    // Util to get the last parameter of a parameter pack
    template <typename T>
    FK_HOST_DEVICE_CNST T last(const T& t) {
        return t;
    }

    template <typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto last(const T& t, const Args&... args) {
        return last(args...);
    }

    template <typename T, typename... Args>
    FK_HOST_DEVICE_CNST T first(const T& t, const Args&... args) {
        return t;
    }

    // Util to insert an element before the last element of a tuple
    template <typename T, typename Tuple>
    FK_HOST_DEVICE_CNST auto insert_before_last_tup(const T& t, const Tuple& args) {
        return tuple_insert<Tuple::size - 1>(t, args);
    }

    template<typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto insert_before_last(const T& t, const Args&... args) {
        return tuple_insert<sizeof...(Args) - 1>(t, Tuple<Args...>{args...});
    }
} // namespace fused kernel
