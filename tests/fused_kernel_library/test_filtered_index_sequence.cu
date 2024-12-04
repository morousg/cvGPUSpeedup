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

#include "tests/main.h"

#include <tests/fkTestsCommon.h>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>

namespace fk {

    struct NotUnaryRestriction {
        template <typename Type>
        static constexpr __host__ __device__ bool complies() {
            using NotUnary =
                TypeList<ReadType, ReadBackType, BinaryType, TernaryType, MidWriteType, WriteType>;
            if constexpr (one_of_v<Type, NotUnary>) {
                return true;
            } else {
                return false;
            }
        }
    };

} // namespace fk

int launch() {
    using ListToCheck =
        fk::TypeList<fk::ReadType, fk::BinaryType, fk::UnaryType, fk::TernaryType,
                     fk::UnaryType, fk::MidWriteType, fk::BinaryType, fk::WriteType>;

    using IndexList = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, ListToCheck>;

    constexpr IndexList indexList;

    static_assert(IndexList::size() == 6, "The index list does not have the expected size");
    
    constexpr size_t var = fk::get_index_f<0>(indexList);

    static_assert(var == 0, "Incorrect index");
    static_assert(fk::get_index<1, IndexList> == 1, "Incorrect index");
    static_assert(fk::get_index<2, IndexList> == 3, "Incorrect index");
    static_assert(fk::get_index<3, IndexList> == 5, "Incorrect index");
    static_assert(fk::get_index<4, IndexList> == 6, "Incorrect index");
    static_assert(fk::get_index<5, IndexList> == 7, "Incorrect index");

    return 0;
}