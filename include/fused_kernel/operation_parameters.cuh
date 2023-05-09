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

// We need to add O so that the compiler knows the otput type of Operator before knowing Operator.
// It can not deduce O from Operator
template <typename Operator, typename I2, typename O=I2>
struct binary_operation_scalar {
    const I2 scalar;
};

/*template <typename Operator, typename T>
struct binary_operation_pointer {
    T* pointer;
    T temp_register[4];
};*/

// We need to add O so that the compiler knows the otput type of Operator before knowing Operator.
// It can not deduce O from Operator
template <typename Operator, typename O>
struct unary_operation_scalar {};

}