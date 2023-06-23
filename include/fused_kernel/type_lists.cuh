/* Copyright 2023 Oscar Amoros Huguet
   Copyright 2023 David Del Rio Astorga
   Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

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

#include <stddef.h>

namespace fk { // namespace fused kernel
    /**
     * @struct TypeList
     * @brief Struct to hold a list of types, and be able to work with them at compile time.
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename... Types>
    struct TypeList {};

    /**
     * @struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>, TypeList<Args4...>>
     * @brief Struct to fuse 4 TypeList into a single one.
     *
     * The expansion results into a single struct that holds all the types.
     */
    template<typename... Args1, typename... Args2, typename... Args3, typename... Args4>
    struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>, TypeList<Args4...>> {
        using type = TypeList<Args1..., Args2..., Args3..., Args4...>;
    };

    template <typename... Args>
    struct one_of {};

    template <typename T, typename... U>
    struct one_of<T, TypeList<U...>> {
        static constexpr int value = std::disjunction_v<std::is_same<T,U>...>;
    };

    template <typename T, typename TypeList_t>
    constexpr bool one_of_v = one_of<T, TypeList_t>::value;

    /**
     * @struct TypeIndex
     * @brief Struct to find at compile time, the index in which the type T is found
     * in the TypeList TypeList_t.
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename T, typename TypeList_t>
    struct TypeIndex;

    /**
     * @struct TypeIndex<T, TypeList<T, Types...>>
     * @brief TypeIndex especialization that implements the case when T is
     * the same type as the first type in TypeList.
     *
     * This the stop condition of the recursive algorithm.
     */
    template <typename T, typename... Types>
    struct TypeIndex<T, TypeList<T, Types...>> {
        static constexpr std::size_t value = 0;
    };

    /**
     * @struct TypeIndex<T, TypeList<U, Types...>>
     * @brief TypeIndex especialization that implements the case when T is
     * not the same type as the first type in TypeList.
     *
     * If T is not the same type as the first Type in TypeList, U, then we define value to be 1 + 
     * whatever is expanded by TypeIndex<T, TypeList<Types...>>::value which will evaluate the 
     * TypeList minus U type.
     */
    template <typename T, typename U, typename... Types>
    struct TypeIndex<T, TypeList<U, Types...>> {
        static_assert(one_of<T, TypeList<U, Types...>>::value == true, "The type is not on the type list");
        static constexpr std::size_t value = 1 + TypeIndex<T, TypeList<Types...>>::value;
    };

    /**
     * \var constexpr size_t TypeIndex_v
     * \brief Template variable that will hold the result of expanding TypeIndex<T, TypeList_t>::value
     */
    template <typename T, typename TypeList_t>
    constexpr size_t TypeIndex_v = TypeIndex<T, TypeList_t>::value;

    // Obtain the type found in the index Idx, in TypeList
    template <std::size_t n, typename... TypeList_t>
    struct TypeAt;

    template <typename Head, typename... Tail>
    struct TypeAt<0, TypeList<Head, Tail...>> {
        using type = Head;
    };

    template <std::size_t n, typename Head, typename... Tail>
    struct TypeAt<n, TypeList<Head, Tail...>> {
        using type = typename TypeAt<n - 1, TypeList<Tail...>>::type;
    };

    template <std::size_t n, typename TypeList_t>
    using TypeAt_t = typename TypeAt<n, TypeList_t>::type;

    // Find the index of T in TypeList1 and obtain the tyoe for that index
    // in TypeList2. All this at compile time. This can be used when you want to automatically derive
    // a type from another type.
    template <typename T, typename TypeList1, typename TypeList2>
    struct EquivalentType {
        static_assert(one_of_v<T, TypeList1>, "The type is not in the first list");
        using type = TypeAt_t<TypeIndex_v<T, TypeList1>, TypeList2>;
    };

    template <typename T, typename TypeList1, typename TypeList2>
    using EquivalentType_t = typename EquivalentType<T, TypeList1, TypeList2>::type;

    template <typename T, std::size_t SIZE>
    struct Array {
        T at[SIZE];
    };

    template <typename T, int BATCH, typename... Vals>
    FK_HOST_DEVICE_CNST Array<T, BATCH> make_array(Vals... pars) {
        static_assert(sizeof...(Vals) == BATCH, "Too many or too few elements for the array size.");
        return { {pars...} };
    }

    template<unsigned... args>
    struct IndexArray {
        static const uint at[sizeof...(args)];
    };

    template<unsigned... args>
    const uint IndexArray<args...>::at[sizeof...(args)] = { args... };

    template<size_t N, template<size_t> class F, unsigned... args>
    struct generate_array_impl {
        using result = typename generate_array_impl<N - 1, F, F<N>::value, args...>::result;
    };

    template<template<size_t> class F, unsigned... args>
    struct generate_array_impl<0, F, args...> {
        using result = IndexArray<F<0>::value, args...>;
    };

    template<size_t N, template<size_t> class F>
    struct generate_array {
        using result = typename generate_array_impl<N - 1, F>::result;
    };

}; // namespace fused kernel