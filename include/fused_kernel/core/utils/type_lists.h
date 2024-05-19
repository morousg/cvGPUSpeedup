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
#include <fused_kernel/core/utils/utils.h>

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

    template<typename... Args1, typename... Args2, typename... Args3,
             typename... Args4, typename... Args5, typename... Args6>
    struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>,
                    TypeList<Args4...>, TypeList<Args5...>, TypeList<Args6...>> {
        using type = TypeList<Args1..., Args2..., Args3..., Args4..., Args5..., Args6...>;
    };

    template<typename... Types>
    struct TypeListCat{};

    template<typename... Args1, typename... Args2>
    struct TypeListCat<TypeList<Args1...>, TypeList<Args2...>> {
        using type = TypeList<Args1..., Args2...>;
    };

    template <typename TypeList1, typename TypeList2>
    using TypeListCat_t = typename TypeListCat<TypeList1, TypeList2>::type;

    template <typename... Args>
    struct one_of {};

    template <typename T, typename... U>
    struct one_of<T, TypeList<U...>> {
        static constexpr int value = std::disjunction_v<std::is_same<T,U>...>;
    };

    template <typename T, typename TypeList_t>
    constexpr bool one_of_v = one_of<T, TypeList_t>::value;

    template <typename TypeListA, typename TypeListB>
    struct one_of_or {};

    template <typename... TypesA, typename... TypesB>
    struct one_of_or<TypeList<TypesA...>, TypeList<TypesB...>> {
        static constexpr bool value = (one_of_v<TypesA, TypeList<TypesB...>> || ...);
    };

    template <typename TypeListA, typename TypeListB>
    constexpr bool one_of_or_v = one_of_or<TypeListA, TypeListB>::value;

    template <typename TypeListA, typename TypeListB>
    struct one_of_and {};

    template <typename... TypesA, typename... TypesB>
    struct one_of_and<TypeList<TypesA...>, TypeList<TypesB...>> {
        static constexpr bool value = (one_of_v<TypesA, TypeList<TypesB...>> && ...);
    };

    template <typename TypeListA, typename TypeListB>
    constexpr bool one_of_and_v = one_of_and<TypeListA, TypeListB>::value;

    /**
     * @struct EnumType
     * @brief Struct to convert an enum value into a type
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename Enum, Enum value>
    struct EnumType {};

    template <typename Enum, Enum value>
    using E_t = EnumType<Enum, value>;

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
    template <int n, typename... TypeList_t>
    struct TypeAt;

    template <typename Head, typename... Tail>
    struct TypeAt<0, TypeList<Head, Tail...>> {
        using type = Head;
    };

    template <typename Head, typename... Tail>
    struct TypeAt<-1, TypeList<Head, Tail...>> {
        using type = typename TypeAt<sizeof...(Tail)-1, TypeList<Tail...>>::type;
    };

    template <typename Head>
    struct TypeAt<-1, TypeList<Head>> {
        using type = Head;
    };

    template <int n, typename Head, typename... Tail>
    struct TypeAt<n, TypeList<Head, Tail...>> {
        using type = typename TypeAt<n - 1, TypeList<Tail...>>::type;
    };

    template <int n, typename TypeList_t>
    using TypeAt_t = typename TypeAt<n, TypeList_t>::type;

    template <typename... Types>
    using FirstType_t = TypeAt_t<0, TypeList<Types...>>;

    template <typename... Types>
    using LastType_t = TypeAt_t<sizeof...(Types)-1, TypeList<Types...>>;

    template <typename... Types>
    using FirstDeviceFunctionInputType_t = typename FirstType_t<Types...>::Operation::InputType;

    template <typename... Types>
    using LastDeviceFunctionOutputType_t = typename LastType_t<Types...>::Operation::OutputType;

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

    template <typename T, size_t BATCH, typename... Types>
    FK_HOST_DEVICE_CNST Array<T, BATCH> make_array(Types... pars) {
        static_assert(sizeof...(Types) == BATCH, "Too many or too few elements for the array size.");
        static_assert(std::disjunction_v<std::is_same<T, Types>...>, "All the types should be the same");
        return { {pars...} };
    }

    template<typename T, typename... Ts>
    constexpr bool all_types_are_ = std::conjunction_v<std::is_same<T, Ts>...>;

    template <std::size_t Index, typename T, typename... Types>
    struct InsertType {};

    template <typename T>
    struct InsertType<0, T> {
        using type = TypeList<T>;
    };

    template <std::size_t Index, typename T, typename Head>
    struct InsertType<Index, T, TypeList<Head>> {
        using type = std::conditional_t<Index == 0,
            TypeList<T, Head>,
            TypeList<Head, T>
        >;
    };

    template <std::size_t Index, typename T, typename Head, typename... Tail>
    struct InsertType<Index, T, TypeList<Head, Tail...>> {
        using type = std::conditional_t<Index == 0,
                                        TypeList<T, Head, Tail...>,
                                        TypeListCat_t<TypeList<Head>, typename InsertType<Index - 1, T, TypeList<Tail...>>::type>
                                       >;
    };

    template <std::size_t Index, typename T, typename... Types>
    using InsertType_t = typename InsertType<Index, T, Types...>::type;
}; // namespace fused kernel