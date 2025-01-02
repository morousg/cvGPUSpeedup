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

    template <typename OperationORInstantiableOperation>
    constexpr bool isReadType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, ReadType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isReadBackType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, ReadBackType>;

    using ReadTypeList = TypeList<ReadType, ReadBackType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isAnyReadType = one_of_v<typename OperationORInstantiableOperation::InstanceType, ReadTypeList>;

    template <typename OperationORInstantiableOperation, typename = void>
    struct is_any_read_type : std::false_type {};

    template <typename OperationORInstantiableOperation>
    struct is_any_read_type<OperationORInstantiableOperation, std::enable_if_t<isAnyReadType<OperationORInstantiableOperation>, void>> : std::true_type {};

    template <typename OperationORInstantiableOperation>
    constexpr bool isUnaryType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, UnaryType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isBinaryType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, BinaryType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isTernaryType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, TernaryType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isWriteType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, WriteType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isMidWriteType = std::is_same_v<typename OperationORInstantiableOperation::InstanceType, MidWriteType>;

    using ComputeTypeList = TypeList<UnaryType, BinaryType, TernaryType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isComputeType = one_of_v<typename OperationORInstantiableOperation::InstanceType, ComputeTypeList>;

    using WriteTypeList = TypeList<WriteType, MidWriteType>;

    template <typename OperationORInstantiableOperation>
    constexpr bool isAnyWriteType = one_of_v<typename OperationORInstantiableOperation::InstanceType, WriteTypeList>;

    template <typename InstantiableOperation>
    using GetInputType_t = typename InstantiableOperation::Operation::InputType;

    template <typename InstantiableOperation>
    using GetOutputType_t = typename InstantiableOperation::Operation::OutputType;

    template <typename InstantiableOperation>
    FK_HOST_DEVICE_CNST GetOutputType_t<InstantiableOperation> compute(const GetInputType_t<InstantiableOperation>& input,
                                                                       const InstantiableOperation& instantiableOperation) {
        static_assert(isComputeType<InstantiableOperation>, "Function compute only works with InstantiableOperation InstanceTypes of the group ComputeTypeList");
        if constexpr (isUnaryType<InstantiableOperation>) {
            return InstantiableOperation::Operation::exec(input);
        } else {
            return InstantiableOperation::Operation::exec(input, instantiableOperation);
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
