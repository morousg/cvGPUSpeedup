/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_OPERATION_TYPES
#define FK_OPERATION_TYPES

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/utils/type_lists.h>

namespace fk {
    struct ReadType {};
    struct ReadBackType {};
    struct UnaryType {};
    struct BinaryType {};
    struct TernaryType {};
    struct MidWriteType {};
    struct WriteType {};

    template <typename OpORIOp>
    constexpr bool isReadType = std::is_same_v<typename OpORIOp::InstanceType, ReadType>;

    template <typename OpORIOp>
    constexpr bool isReadBackType = std::is_same_v<typename OpORIOp::InstanceType, ReadBackType>;

    using ReadTypeList = TypeList<ReadType, ReadBackType>;

    template <typename OpORIOp>
    constexpr bool isAnyReadType = one_of_v<typename OpORIOp::InstanceType, ReadTypeList>;

    template <typename OpORIOp, typename = void>
    struct is_any_read_type : std::false_type {};

    template <typename OpORIOp>
    struct is_any_read_type<OpORIOp, std::enable_if_t<std::is_same_v<typename OpORIOp::InstanceType, ReadType> ||
                                                      std::is_same_v<typename OpORIOp::InstanceType, ReadBackType>, void>> : std::true_type {};

    template <typename OpORIOp>
    constexpr bool isUnaryType = std::is_same_v<typename OpORIOp::InstanceType, UnaryType>;

    template <typename OpORIOp>
    constexpr bool isBinaryType = std::is_same_v<typename OpORIOp::InstanceType, BinaryType>;

    template <typename OpORIOp>
    constexpr bool isTernaryType = std::is_same_v<typename OpORIOp::InstanceType, TernaryType>;

    template <typename OpORIOp>
    constexpr bool isWriteType = std::is_same_v<typename OpORIOp::InstanceType, WriteType>;

    template <typename OpORIOp>
    constexpr bool isMidWriteType = std::is_same_v<typename OpORIOp::InstanceType, MidWriteType>;

    using ComputeTypeList = TypeList<UnaryType, BinaryType, TernaryType>;

    template <typename OpORIOp>
    constexpr bool isComputeType = one_of_v<typename OpORIOp::InstanceType, ComputeTypeList>;

    using WriteTypeList = TypeList<WriteType, MidWriteType>;

    template <typename OpORIOp>
    constexpr bool isAnyWriteType = one_of_v<typename OpORIOp::InstanceType, WriteTypeList>;

    template <typename IOp>
    using GetInputType_t = typename IOp::Operation::InputType;

    template <typename IOp>
    using GetOutputType_t = typename IOp::Operation::OutputType;

    template <typename IOp>
    FK_HOST_DEVICE_CNST GetOutputType_t<IOp> compute(const GetInputType_t<IOp>& input,
                                                                       const IOp& instantiableOperation) {
        static_assert(isComputeType<IOp>, "Function compute only works with IOp InstanceTypes of the group ComputeTypeList");
        if constexpr (isUnaryType<IOp>) {
            return IOp::Operation::exec(input);
        } else {
            return IOp::Operation::exec(input, instantiableOperation);
        }
    }

    template <typename... OperationsOrInstantiableOperations>
    constexpr bool allUnaryTypes = and_v<isUnaryType<OperationsOrInstantiableOperations>...>;

    template <typename Enabler, typename... OperationsOrInstantiableOperations>
    struct are_all_unary_types : std::false_type {};

    template <typename... OperationsOrInstantiableOperations>
    struct are_all_unary_types<std::enable_if_t<allUnaryTypes<OperationsOrInstantiableOperations...>>,
                               OperationsOrInstantiableOperations...> : std::true_type {};

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneWriteType = and_v<(!isWriteType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneMidWriteType = and_v<(!isMidWriteType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneAnyWriteType = and_v<(!isAnyWriteType<OperationORInstantiableOperation>)...>;
} // namespace fk

#endif
