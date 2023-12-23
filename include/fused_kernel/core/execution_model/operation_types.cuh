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

    struct ReadType {};
    struct WriteType {};
    struct UnaryType {};
    struct BinaryType {};
    struct MidWriteType {};

    template <typename Operation>
    constexpr bool isReadOperation = std::is_same_v<typename Operation::InstanceType, ReadType>;

} // namespace fk
