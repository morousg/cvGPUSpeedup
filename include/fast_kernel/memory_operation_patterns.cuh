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

template <int NPtr, typename Operator, typename T, typename Enabler=void>
struct memory_read_iterpolated_N;

template <typename Operator, typename T>
struct memory_read_iterpolated_N<1, Operator, T, std::enable_if_t<true>> {
    const RawPtr<_2D,T> ptr;
    const float fx;
    const float fy;
    const uint target_width;
    const uint target_height;
};

// TODO: maybe use always this version with NPtr = 1 for the current 2D only version
template <int NPtr, typename Operator, typename T>
struct memory_read_iterpolated_N<NPtr, Operator, T, std::enable_if_t<(NPtr>1)>> {
    RawPtr<_2D,T> ptr[NPtr];
    float fx[NPtr];
    float fy[NPtr];
    uint target_width;
    uint target_height;
};

template <ND D, typename Operator, typename T>
struct memory_write_scalar {
    RawPtr<D,T> ptr;
};

template <ND D, typename Operator, typename T, typename Enabler=void>
struct split_write_scalar {};

template <ND D, typename Operator, typename T>
struct split_write_scalar<D, Operator, T, typename std::enable_if_t<CN(T) == 2>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
};

template <ND D, typename Operator, typename T>
struct split_write_scalar<D, Operator, T, typename std::enable_if_t<CN(T) == 3>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
};

template <ND D, typename Operator, typename T>
struct split_write_scalar<D, Operator, T, typename std::enable_if_t<CN(T) == 4>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
    RawPtr<D, decltype(T::w)> w;
};

template <typename Operator, typename T>
struct split_write_tensor {
    RawPtr<_3D, T> t;
};

}