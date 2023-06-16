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

#include <tuple>
#include <stddef.h>

namespace fk { // namespace fused kernel

    template <typename... Types>
    struct TypeList {};

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

    template <typename T, typename TypeList_t>
    struct TypeIndex;

    template <typename T, typename... Types>
    struct TypeIndex<T, TypeList<T, Types...>> {
        static_assert(one_of<T, TypeList<T, Types...>>::value == true, "The type is not on the type list");
        static constexpr std::size_t value = 0;
    };

    template <typename T, typename U, typename... Types>
    struct TypeIndex<T, TypeList<U, Types...>> {
        static_assert(one_of<T, TypeList<U, Types...>>::value == true, "The type is not on the type list");
        static constexpr std::size_t value = 1 + TypeIndex<T, TypeList<Types...>>::value;
    };

    template <typename T, typename TypeList_t>
    constexpr size_t TypeIndex_v = TypeIndex<T, TypeList_t>::value;

    template <int Idx, typename TypeList_t>
    struct TypeFromIndex;

    template <int Idx, typename... Types>
    struct TypeFromIndex<Idx, TypeList<Types...>> {
        static_assert(Idx < sizeof...(Types), "Index out of range");
        using type = std::tuple_element_t<Idx, std::tuple<Types...>>;
    };

    template <int Idx, typename TypeList_t>
    using TypeFromIndex_t = typename TypeFromIndex<Idx, TypeList_t>::type;

    template <typename T, typename TypeList1, typename TypeList2>
    struct EquivalentType {
        static_assert(one_of_v<T, TypeList1>, "The type is not in the first list");
        using type = TypeFromIndex_t<TypeIndex_v<T, TypeList1>, TypeList2>;
    };

    template <typename T, typename TypeList1, typename TypeList2>
    using EquivalentType_t = typename EquivalentType<T, TypeList1, TypeList2>::type;

}; // namespace fused kernel