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

#pragma once

#include <fused_kernel/core/utils/utils.h>

namespace fk {
    struct Point {
        uint x;
        uint y;
        uint z;
        FK_HOST_DEVICE_CNST Point(const uint x_ = 0, const uint y_ = 0, const uint z_ = 0) : x(x_), y(y_), z(z_) {}
    };
} // namespace fk
