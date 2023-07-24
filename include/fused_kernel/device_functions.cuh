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

#include <vector_types.h>

namespace fk { // namespace FusedKernel
    // generic operation structs
    template <typename Operation_t>
    struct ReadDeviceFunction {
        typename Operation_t::ParamsType params;
        dim3 activeThreads;
        using Operation = Operation_t;
    };

    template <typename Operation_t>
    struct BinaryDeviceFunction {
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
    };

    template <typename Operation_t>
    struct UnaryDeviceFunction {
        using Operation = Operation_t;
    };

    template <typename Operation_t>
    struct MidWriteDeviceFunction {
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
    };

    template <typename Operation_t>
    struct WriteDeviceFunction {
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
    };
} // namespace FusedKernel
