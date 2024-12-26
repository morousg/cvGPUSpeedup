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

#ifndef FK_STATIC_LOOP
#define FK_STATIC_LOOP

#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    template <typename Operation, int ITERATIONS>
    struct StaticLoop {
        using InputType = typename Operation::InputType;
        using OutputType = typename Operation::OutputType;
        using ParamsType = typename Operation::ParamsType;
        using InstanceType = BinaryType;

        private:
        template <int ITERATION>
        FK_DEVICE_FUSE OutputType helper_exec(const InputType& input, const ParamsType& params) {
            if constexpr (ITERATION + 1 < ITERATIONS) {
                return helper_exec<ITERATION + 1>(Operation::exec(input, params), params);
            } else {
                return input;
            }
        }

        public:
        FK_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return helper_exec<0>(Operation::exec(input, params), params);
        }
        using InstantiableType = Binary<StaticLoop<Operation, ITERATIONS>>;
        DEFAULT_BINARY_BUILD
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
