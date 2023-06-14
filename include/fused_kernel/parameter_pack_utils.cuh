/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace fk { // namespace fused kernel

    template <typename F, typename Tuple, size_t... I>
    __host__ __device__ __forceinline__ constexpr
    auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
        -> decltype(std::forward<F>(f)(thrust::get<I>(std::forward<Tuple>(t))...))
    {
        return std::forward<F>(f)(thrust::get<I>(std::forward<Tuple>(t))...);
    }

    template <typename F, typename Tuple>
    __host__ __device__ __forceinline__ constexpr
    auto apply(F&& f, Tuple&& t)
        -> decltype(apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<thrust::tuple_size<typename std::decay<Tuple>::type>::value>()))
    {
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<thrust::tuple_size<typename std::decay<Tuple>::type>::value>());
    }

    // Struct to hold a parameter pack, and be able to pass it arround
    template <typename... Args>
    struct OperationSequence {
        thrust::tuple<const Args...> args;
    };

    // Function that fills the OperationSequence struct, from a parameter pack
    template <typename... operations>
    inline constexpr auto buildOperationSequence(const operations&... ops) {
        return OperationSequence<operations...> {{ops...}};
    }

    // Util to get the last parameter of a parameter pack
    template <typename T>
    __device__ __forceinline__ constexpr T last(const T& t) {
        return t;
    }

    template <typename T, typename... Args>
    __device__ __forceinline__ constexpr auto last(const T& t, const Args&... args) {
        return last(args...);
    }

    template <typename... Args>
    __host__ __device__ __forceinline__ constexpr auto tuple_cat(Args&&... args) {
        return thrust::make_tuple(std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    __host__ __device__ __forceinline__ constexpr auto insert_before_last_tup(const T& t, const thrust::tuple<Args...>& args) {
        return args;
    }

    template<typename T, typename... Args>
    __host__ __device__ __forceinline__ constexpr auto insert_before_last(const T& t, const Args&... args) {
        thrust::tuple<const Args...> paramPack{args...};
        return insert_before_last_tup(t, paramPack);
    }

} // namespace fused kernel
