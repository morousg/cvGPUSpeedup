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

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/tuple.cuh>

#include <array>

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
        Tuple<DeviceFunctionTypes...> deviceFunctions;
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

    // Util to insert an element before the last element of a tuple
    template <typename T, typename Tuple>
    FK_HOST_DEVICE_CNST auto insert_before_last_tup(const T& t, const Tuple& args) {
        return tuple_insert<Tuple::size - 1>(t, args);
    }

    template<typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto insert_before_last(const T& t, const Args&... args) {
        return tuple_insert<sizeof...(Args) - 1>(t, Tuple<Args...>{args...});
    }

    template <typename TransformType, typename SourceType, size_t NElems, int... Idx, typename... ExtraParams>
    constexpr inline std::array<typename TransformType::OutputType, NElems>
        static_transform_helper( const int& usedPlanes,
                                 const std::array<SourceType, NElems>& srcArray,
                                 const std::integer_sequence<int, Idx...>&,
                                 const ExtraParams&... extParams) {
        return { (TransformType::template transform<Idx>(usedPlanes, srcArray[Idx], extParams...))... };
    }

    template <typename TransformType, typename SourceType, size_t NElems, typename... ExtraParams>
    constexpr inline std::array<typename TransformType::OutputType, NElems>
        static_transform(const int& usedPlanes, const std::array<SourceType, NElems>& srcArray, const ExtraParams&... extParams) {
        return static_transform_helper<TransformType>(usedPlanes, srcArray, std::make_integer_sequence<int, NElems>{}, extParams...);
    }

    template <typename FirstType, typename SecondType>
    struct GetFirst {
        using OutputType = FirstType;
        template <int Idx>
        static constexpr inline FirstType transform(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.first;
        }
    };

    template <typename FirstType, typename SecondType>
    struct GetSecond {
        using OutputType = SecondType;
        template <int Idx>
        static constexpr inline SecondType transform(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.second;
        }
    };

    template <typename FT, typename ST, size_t NElems>
    constexpr inline std::array<FT, NElems>
        static_transform_get_first(const std::array<std::pair<FT, ST>, NElems>& srcArray) {
        return static_transform_helper<GetFirst<FT,ST>>(NElems, srcArray, std::make_integer_sequence<int, NElems>{});
    }

    template <typename FT, typename ST, size_t NElems>
    constexpr inline std::array<ST, NElems>
        static_transform_get_second(const std::array<std::pair<FT, ST>, NElems>& srcArray) {
        return static_transform_helper<GetSecond<FT, ST>>(NElems, srcArray, std::make_integer_sequence<int, NElems>{});
    }
} // namespace fk
