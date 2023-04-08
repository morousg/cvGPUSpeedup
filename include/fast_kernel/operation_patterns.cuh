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
#include "cuda_utils.cuh"
#include "operation_types.cuh"

namespace fk {

template <typename Operator, typename I1, typename I2, typename O>
struct _binary_operation_scalar {
    I2 scalar;
    Operator nv_operator;
};
template <typename Operator, typename I1, typename I2 = I1, typename O = I1>
using binary_operation_scalar = _binary_operation_scalar<Operator, I1, I2, O>;

template <typename Operator, typename I1, typename I2, typename O>
struct _binary_operation_pointer {
    I2* pointer;
    Operator nv_operator;
    I2 temp_register[4];
};
template <typename Operator, typename I1, typename I2 = I1, typename O = I1>
using binary_operation_pointer = _binary_operation_pointer<Operator, I1, I2, O>;

template <typename Operator, typename I, typename O>
struct _unary_operation_scalar {
    Operator nv_operator;
};
template <typename Operator, typename I, typename O>
using unary_operation_scalar = _unary_operation_scalar<Operator, I, O>;

}