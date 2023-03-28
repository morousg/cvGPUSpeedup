/* Copyright 2023 Oscar Amoros Huguet

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

#include "cuda_utils.h"

#define NUM_COMPONENTS(some_type) (sizeof(some_type)/sizeof(decltype(some_type::x)))

// Automagically making any CUDA vector type from a template type
// It will not compile if you try to do bad things. The number of elements
// need to conform to T, and the type of the elements will always be casted.
template <typename T, typename... Numbers>
inline constexpr __device__ __host__ T make_(Numbers... pack) {
    return {static_cast<decltype(T::x)>(pack)...};
}
