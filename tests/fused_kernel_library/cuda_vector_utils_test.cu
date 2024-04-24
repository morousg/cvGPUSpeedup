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

#include <fused_kernel/core/utils/cuda_vector_utils.h>

#include "tests/main.h"

int launch() {
    static_assert(BasicType<float>, "Error in test of BasicType");
    static_assert(!BasicType<uchar4>, "Error in test of BasicType");
    static_assert(AllBasicTypes<int, float, char, uchar>, "Error in test of AllBasicTypes");
    static_assert(!AllBasicTypes<int, float, float1, uchar>, "Error in test of AllBasicTypes");
    static_assert(AllBasicTypesInList<StandardTypes>, "Error in test of AllBasicTypesInList");
    static_assert(!AllBasicTypesInList<VAll>, "Error in test of AllBasicTypesInList");
    static_assert(AllBasicTypesInList<TypeList<float, int>>, "Error in test of AllBasicTypesInList");
    static_assert(!AllBasicTypesInList<TypeList<float, int, uint1>>, "Error in test of AllBasicTypesInList");
    static_assert(CUDAType<float3>, "Error in test of CUDAType");
    static_assert(!CUDAType<uchar>, "Error in test of CUDAType");
    static_assert(AllCUDATypesInList<TypeList<float3, int2>>, "Error in test of AllCUDATypesInList");
    static_assert(AllCUDATypesInList<VThree>, "Error in test of AllCUDATypesInList");
    static_assert(!AllCUDATypesInList<TypeList<uchar, int3>>, "Error in test of AllCUDATypesInList");

    return 0;
}
