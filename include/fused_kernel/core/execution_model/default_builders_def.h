/* Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef BUILDERS
#define BUILDERS

#define DEFAULT_READ_BATCH_BUILD \
template <size_t Idx, typename Array> \
FK_HOST_FUSE auto get_element_at_index(const Array& array) -> decltype(array[Idx]) { \
    return array[Idx]; \
} \
template <size_t Idx, typename... Arrays> \
FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) { \
    return InstantiableType::Operation::build(get_element_at_index<Idx>(arrays)...); \
} \
template <size_t... Idx, typename... Arrays> \
FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, const Arrays&... arrays) { \
    using OutputArrayType = decltype(InstantiableType::Operation::build(std::declval<typename Arrays::value_type>()...)); \
    return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... }; \
} \
template <size_t BATCH, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE std::array<InstantiableType, BATCH> \
build_batch(const std::array<FirstType, BATCH>& firstInstance, const ArrayTypes&... arrays) { \
    static_assert(allArraysSameSize_v<BATCH, std::array<FirstType, BATCH>, ArrayTypes...>, "Not all arrays have the same size as BATCH"); \
    return build_helper_generic(std::make_index_sequence<BATCH>(), firstInstance, arrays...); \
}

#define DEFAULT_READ_BUILD \
static constexpr __host__ __forceinline__ auto build(const ParamsType& params) { \
    return InstantiableType{ {params} }; \
} \
static constexpr __host__ __forceinline__ \
auto build_source(const ParamsType& params) { \
    using OutputType = decltype(make_source(std::declval<InstantiableType>(), std::declval<ActiveThreads>())); \
    return OutputType{ {params}, {num_elems_x(Point(), params), num_elems_y(Point(), params), num_elems_z(Point(), params)} }; \
}

#define DEFAULT_READBACK_BUILD \
static constexpr __host__ __forceinline__ \
auto build(const ParamsType& params, const BackFunction_& backFunction) { \
    return InstantiableType{ { params, backFunction } }; \
} \
static constexpr __host__ __forceinline__ auto build_source(const ParamsType& params, const BackFunction_& backFunction) { \
    using OutputType = decltype(make_source(std::declval<InstantiableType>(), std::declval<ActiveThreads>())); \
    return OutputType{ { params, backFunction }, \
        { num_elems_x(Point(), params, backFunction), \
          num_elems_y(Point(), params, backFunction), \
          num_elems_z(Point(), params, backFunction) } }; \
}

#define DEFAULT_UNARY_BUILD \
static constexpr __host__ __forceinline__ InstantiableType build() { \
    return InstantiableType{}; \
}

#define DEFAULT_BINARY_BUILD \
static constexpr __host__ __forceinline__ InstantiableType build(const ParamsType& params) { \
    return InstantiableType{ {params} }; \
}

#define DEFAULT_TERNARY_BUILD \
static constexpr __host__ __forceinline__ InstantiableType build(const ParamsType& params, const BackFunction& backFunction) { \
    return InstantiableType{ {params, backFunction} }; \
}

#define DEFAULT_WRITE_BUILD \
static constexpr __host__ __forceinline__ InstantiableType build(const ParamsType& params) { \
    return InstantiableType{ {params} }; \
}

#endif
