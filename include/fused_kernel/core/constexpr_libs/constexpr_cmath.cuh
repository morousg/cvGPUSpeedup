/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_CONSTEXPR_CMATH
#define FK_CONSTEXPR_CMATH

#include <fused_kernel/core/utils/utils.h>

#include <type_traits>
#include <limits>

namespace cxp {

    template <typename T>
    FK_HOST_DEVICE_CNST bool isnan(T x) {
        return x != x;
    }

    template <typename T>
    FK_HOST_DEVICE_CNST bool isinf(T x) {
        return x == x && x != T(0) && x + x == x;
    }

    template<typename T>
    FK_HOST_DEVICE_CNST T round(T x) {
        static_assert(std::is_floating_point<T>::value, "Input must be a floating-point type");

        if (isnan(x) || isinf(x)) {
            return x;
        }
        // Casted to int instead of long long, because long long is very slow on GPU
        return (x > T(0))
            ? static_cast<T>(static_cast<int>(x + T(0.5)))
            : static_cast<T>(static_cast<int>(x - T(0.5)));
    }

} // namespace cxp

#endif
