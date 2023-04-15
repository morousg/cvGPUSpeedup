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

namespace fk {

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
struct binary_operation_scalar {
    const I2 scalar;
};

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
struct binary_operation_pointer {
    I2* pointer;
    I2 temp_register[4];
};

template <typename Operator, typename I, typename O>
struct unary_operation_scalar {};

}