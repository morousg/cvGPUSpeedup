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

#define DEFAULT_READ_BUILD \
static constexpr __host__ __forceinline__ InstantiableType build(const ParamsType& params) { \
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
