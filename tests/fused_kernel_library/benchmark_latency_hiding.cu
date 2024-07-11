/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/fkTestsCommon.h>

#include "tests/main.h"

constexpr size_t NUM_EXPERIMENTS = 30;
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 100;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 1920 * 1080;

template <typename T, typename... DFs>
int testLatencyHiding(const DFs&... deviceFunctions) {
    START_CVGS_BENCHMARK

}

int launch() {


    return 0;
}