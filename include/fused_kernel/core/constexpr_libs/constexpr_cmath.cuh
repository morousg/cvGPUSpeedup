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
#include <fused_kernel/core/utils/type_lists.h>

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

    namespace internal {
        template <typename Type>
        FK_HOST_DEVICE_CNST auto max_helper(const Type& value) {
            return value;
        }
        template <typename FirstType, typename... Types>
        FK_HOST_DEVICE_CNST auto max_helper(const FirstType& firstValue,
                                            const Types&... values) {
            const auto previousMax = max_helper(values...);
            return firstValue >= previousMax ? firstValue : previousMax;
        }
        template <typename Type>
        FK_HOST_DEVICE_CNST auto min_helper(const Type& value) {
            return value;
        }
        template <typename FirstType, typename... Types>
        FK_HOST_DEVICE_CNST auto min_helper(const FirstType& firstValue,
            const Types&... values) {
            const auto previousMax = max_helper(values...);
            return firstValue <= previousMax ? firstValue : previousMax;
        }
    } // namespace internal

    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto max(const FirstType& value, const Types&... values) {
        static_assert(fk::all_types_are_same<FirstType, Types...>, "All types must be the same");
        return internal::max_helper(value, values...);
    }

    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto min(const FirstType& value, const Types&... values) {
        static_assert(fk::all_types_are_same<FirstType, Types...>, "All types must be the same");
        return internal::min_helper(value, values...);
    }

    template <typename T>
    FK_HOST_DEVICE_CNST T abs(const T& x) {
        static_assert(std::is_fundamental_v<T>, "abs does not support non fundamental types");
        if constexpr (std::is_signed_v<T>) {
            if (x == std::numeric_limits<T>::min()) {
                return std::numeric_limits<T>::max();
            }
            return x < T(0) ? -x : x;
        } else {
            return x;
        }
    }

} // namespace cxp

#endif
