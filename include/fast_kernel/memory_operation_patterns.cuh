/* Copyright 2023 Oscar Amoros Huguet

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
#include "memory_operation_types.cuh"

namespace fk {

template <typename Operator, typename T>
struct memory_read_iterpolated {
    const PtrAccessor<T> ptr;
    const float fx;
    const float fy;
    const uint target_width;
    const uint target_height;
};

template <typename Operator, typename O>
struct memory_write_scalar_2D {
    PtrAccessor<O> x;
};

template <typename Operator, typename I, typename Enabler=void>
struct split_write_scalar_2D {};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<CN(I) == 2>> {
    PtrAccessor<decltype(I::x)> x;
    PtrAccessor<decltype(I::y)> y;
};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<CN(I) == 3>> {
    PtrAccessor<decltype(I::x)> x;
    PtrAccessor<decltype(I::y)> y;
    PtrAccessor<decltype(I::z)> z;
};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<CN(I) == 4>> {
    PtrAccessor<decltype(I::x)> x;
    PtrAccessor<decltype(I::y)> y;
    PtrAccessor<decltype(I::z)> z;
    PtrAccessor<decltype(I::w)> w;
};

}