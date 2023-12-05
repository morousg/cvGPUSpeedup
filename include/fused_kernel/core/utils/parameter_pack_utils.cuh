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

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <fused_kernel/core/utils/cuda_utils.cuh>

namespace fk { // namespace fused kernel

    template <typename F, typename Tuple, size_t... I>
    FK_HOST_DEVICE_CNST auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
        -> decltype(std::forward<F>(f)(thrust::get<I>(std::forward<Tuple>(t))...))
    {
        return std::forward<F>(f)(thrust::get<I>(std::forward<Tuple>(t))...);
    }

    template <typename F, typename Tuple>
    FK_HOST_DEVICE_CNST auto apply(F&& f, Tuple&& t)
        -> decltype(apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<thrust::tuple_size<typename std::decay<Tuple>::type>::value>()))
    {
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<thrust::tuple_size<typename std::decay<Tuple>::type>::value>());
    }

    // Struct to hold a parameter pack, and be able to pass it arround
    template <typename... DeviceFunctionTypes>
    struct DeviceFunctionSequence {
        thrust::tuple<const DeviceFunctionTypes...> deviceFunctions;
    };

    // Function that fills the OperationSequence struct, from a parameter pack
    template <typename... DeviceFunctionTypes>
    FK_HOST_DEVICE_CNST auto buildOperationSequence(const DeviceFunctionTypes&... deviceFunctionInstances) {
        return DeviceFunctionSequence<DeviceFunctionTypes...> {{deviceFunctionInstances...}};
    }

    template <typename... DeviceFunctionTypes>
    FK_HOST_DEVICE_CNST auto buildOperationSequence_tup(const thrust::tuple<DeviceFunctionTypes...>& deviceFunctionInstances) {
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

    // Util to concatenate thrust::tuple
    template <typename Tuple1, typename Tuple2, int... I1, int... I2>
    FK_HOST_DEVICE_CNST
    auto tuple_cat_impl(const Tuple1& t1, std::integer_sequence<int, I1...>, const Tuple2& t2, std::integer_sequence<int, I2...>) {
        static_assert(thrust::tuple_size<Tuple1>::value + thrust::tuple_size<Tuple2>::value <= 10,
                      "thrust::tuple max size is 10, you are trying to create a bigger tuple");
        return thrust::make_tuple(thrust::get<I1>(t1)..., thrust::get<I2>(t2)...);
    }

    template <typename Tuple1, typename Tuple2>
    FK_HOST_DEVICE_CNST
    auto tuple_cat(const Tuple1& t1, const Tuple2& t2) {
        return tuple_cat_impl(t1, std::make_integer_sequence<int, thrust::tuple_size<Tuple1>::value>(),
            t2, std::make_integer_sequence<int, thrust::tuple_size<Tuple2>::value>());
    }

    // Util to insert an element before the last element of a tuple
    template <typename T, typename Tuple>
    FK_HOST_DEVICE_CNST auto insert_before_last_tup(const T& t, const Tuple& args) {
        if constexpr (thrust::tuple_size<Tuple>::value == 1) {
            return fk::tuple_cat(thrust::make_tuple(t), args);
        } else {
            const auto [head, tail] = args;
            return fk::tuple_cat(thrust::make_tuple(head), insert_before_last_tup(t, tail));
        }
    }

    template<typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto insert_before_last(const T& t, const Args&... args) {
        return insert_before_last_tup(t, thrust::tuple<const Args...>{args...});
    }

    template <typename T>
    struct is_thrust_tuple : std::false_type {};

    template <typename... Args>
    struct is_thrust_tuple<thrust::tuple<Args...>> : std::true_type {};

    template <typename T>
    constexpr bool is_thrust_tuple_v = is_thrust_tuple<T>::value;
} // namespace fused kernel
