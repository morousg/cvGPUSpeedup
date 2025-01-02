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

#ifndef FK_INSTANTIABLE_DATA_PARALLEL_PATTERNS
#define FK_INSTANTIABLE_DATA_PARALLEL_PATTERNS

#include <cooperative_groups.h>

namespace cooperative_groups {};
namespace cg = cooperative_groups;

#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    struct DPP_UnaryType {};
    struct DPP_BinaryType {};

    template <typename DPP_, typename Enabler = void>
    struct InstantiableDPP;
    
    template <typename DPP_>
    struct InstantiableDPP<DPP_, std::enable_if_t<DPP_::template is<DPP_BinaryType>>> {
        typename DPP_::Details details;
        typename DPP_::Operations operations;
        using InstanceType = DPP_BinaryType;
        using DPP = DPP_;
        template <typename DPPInstanceType>
        static constexpr bool is = DPP::template is<DPPInstanceType>;
        FK_HOST_CNST ActiveThreads getActiveThreads() const {
            return DPP::getActiveThreads(details, operations);
        }
    };

    template <typename DPP_>
    struct InstantiableDPP<DPP_, std::enable_if_t<DPP_::template is<DPP_UnaryType>>> {
        typename DPP_::Operations operations;
        using InstanceType = DPP_UnaryType;
        using DPP = DPP_;
        template <typename DPPInstanceType>
        static constexpr bool is = DPP::template is<DPPInstanceType>;
        FK_HOST_CNST ActiveThreads getActiveThreads() const {
            return DPP::getActiveThreads(operations);
        }
    };

    template <bool THREAD_FUSION, typename... IOps>
    struct BuildTFI {
        using ReadOp = typename FirstType_t<IOps...>::Operation;
        using WriteOp = typename LastType_t<IOps...>::Operation;
        using TFI =
            ThreadFusionInfo<typename ReadOp::ReadDataType,
                             typename WriteOp::WriteDataType,
                             isThreadFusionEnabled<THREAD_FUSION, IOps...>()>;
    };

    template <typename Enabler, bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_;
    
    template <bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_<std::enable_if_t<BuildTFI<THREAD_FUSION, IOps...>::TFI::ENABLED, void>,
                               THREAD_FUSION, IOps...> {
        ActiveThreads activeThreads;
        bool threadDivisible;
        using OpsTupleType = Tuple<IOps...>;
        using TFI = typename BuildTFI<THREAD_FUSION, IOps...>::TFI;
    };

    template <bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_<std::enable_if_t<!BuildTFI<THREAD_FUSION, IOps...>::TFI::ENABLED, void>,
                                THREAD_FUSION, IOps...> {
        using OpsTupleType = Tuple<IOps...>;
        using TFI = typename BuildTFI<THREAD_FUSION, IOps...>::TFI;
    };

    template <bool THREAD_FUSION, typename... IOps>
    using TransformDPPDetails = TransformDPPDetails_<void, THREAD_FUSION, IOps...>;

    template <typename DPPDetails = void>
    struct TransformDPP {
        using Details = DPPDetails;
        using Operations = typename Details::OpsTupleType;
        using InstanceType = DPP_BinaryType;
        template <typename IType>
        static constexpr bool is = std::is_same_v<IType, InstanceType>;

        template <typename T, typename InstantiableOperation, typename... InstantiableOperationTypes>
        FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const InstantiableOperation& df, const InstantiableOperationTypes&... instantiableOperationInstances) {
            if constexpr (InstantiableOperation::template is<WriteType>) {
                return i_data;
            // MidWriteOperation with continuations, based on FusedOperation
            } else if constexpr (InstantiableOperation::template is<MidWriteType> && isMidWriteType<typename InstantiableOperation::Operation>) {
                return InstantiableOperation::Operation::exec(thread, i_data, df);
            } else if constexpr (InstantiableOperation::template is<MidWriteType> && !isMidWriteType<typename InstantiableOperation::Operation>) {
                InstantiableOperation::Operation::exec(thread, i_data, df);
                return i_data;
            } else {
                return operate(thread, compute(i_data, df), instantiableOperationInstances...);
            }
        }

        template <uint IDX, typename TFI, typename InputType, typename... InstantiableOperationTypes>
        FK_DEVICE_FUSE auto operate_idx(const Point& thread, const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
            return operate(thread, TFI::get<IDX>(input), instantiableOperationInstances...);
        }

        template <typename TFI, typename InputType, uint... IDX, typename... InstantiableOperationTypes>
        FK_DEVICE_FUSE auto operate_thread_fusion_impl(std::integer_sequence<uint, IDX...> idx, const Point& thread,
                                                        const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
            return TFI::make(operate_idx<IDX, TFI>(thread, input, instantiableOperationInstances...)...);
        }

        template <typename TFI, typename InputType, typename... InstantiableOperationTypes>
        FK_DEVICE_FUSE auto operate_thread_fusion(const Point& thread, const InputType& input, const InstantiableOperationTypes&... instantiableOperationInstances) {
            if constexpr (TFI::elems_per_thread == 1) {
                return operate(thread, input, instantiableOperationInstances...);
            } else {
                return operate_thread_fusion_impl<TFI>(std::make_integer_sequence<uint, TFI::elems_per_thread>(), thread, input, instantiableOperationInstances...);
            }
        }

        template <typename ReadIOp, typename TFI>
        FK_DEVICE_FUSE auto read(const Point& thread, const ReadIOp& readDF) {
            if constexpr (TFI::ENABLED) {
                return ReadIOp::Operation::exec<TFI::elems_per_thread>(thread, readDF);
            } else {
                return ReadIOp::Operation::exec(thread, readDF);
            }
        }

        template <typename TFI, typename ReadIOp, typename... IOps>
        FK_DEVICE_FUSE 
        void execute_instantiable_operations_helper(const Point& thread, const ReadIOp& readDF,
                                                    const IOps&... iOps) {
            using ReadOperation = typename ReadIOp::Operation;
            using WriteOperation = typename LastType_t<IOps...>::Operation;

            const auto writeDF = ppLast(iOps...);

            if constexpr (TFI::ENABLED) {
                const auto tempI = read<ReadIOp, TFI>(thread, readDF);
                if constexpr (sizeof...(iOps) > 1) {
                    const auto tempO = operate_thread_fusion<TFI>(thread, tempI, iOps...);
                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDF);
                } else {
                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDF);
                }
            } else {
                const auto tempI = read<ReadIOp, TFI>(thread, readDF);
                if constexpr (sizeof...(iOps) > 1) {
                    const auto tempO = operate(thread, tempI, iOps...);
                    WriteOperation::exec(thread, tempO, writeDF);
                } else {
                    WriteOperation::exec(thread, tempI, writeDF);
                }
            }
        }

        template <typename TFI, size_t... Idx, typename... IOps>
        FK_DEVICE_FUSE
        void execute_instantiable_operations(const std::index_sequence<Idx...>&,
                                             const Point& thread,
                                             const Tuple<IOps...>& iOps){
            execute_instantiable_operations_helper<TFI>(thread, get<Idx>(iOps)...);
        }

        public:
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const Details& details,
                                                           const Operations& operations) {
            if constexpr (Details::TFI::ENABLED) {
                return details.activeThreads;
            } else {
                const auto firstOp = get<0>(operations);
                using ReadType = decltype(firstOp);
                return ReadType::getActiveThreads(firstOp);
            }
        }

        FK_DEVICE_FUSE void exec(const Details& details, const Operations& iOps) {
            const cg::thread_block g = cg::this_thread_block();

            const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
            const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
            const uint z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
            const Point thread{ x, y, z };

            using TFI = typename Details::TFI;

            const ActiveThreads activeThreads = getActiveThreads(details, iOps);

            constexpr auto staticIterator = indexSequence<Operations::size>;

            if (x < activeThreads.x && y < activeThreads.y) {
                if constexpr (!TFI::ENABLED) {
                    execute_instantiable_operations<TFI>(staticIterator, thread, iOps);
                } else {
                    if (details.threadDivisible) {
                        execute_instantiable_operations<TFI>(staticIterator, thread, iOps);
                    } else {
                        if (x < activeThreads.x - 1) {
                            execute_instantiable_operations<TFI>(staticIterator, thread, iOps);
                        } else if (x == activeThreads.x - 1) {
                            const uint initialX = x * TFI::elems_per_thread;
                            using ReadOp = typename get_type_t<0, Operations>::Operation;
                            const uint finalX = ReadOp::num_elems_x(thread, get<0>(iOps).params);
                            uint currentX = initialX;
                            while (currentX < finalX) {
                                const Point currentThread{ currentX , thread.y, thread.z };
                                using ReadIT = typename get_type_t<0, Operations>::Operation::ReadDataType;
                                using WriteOT = typename decltype(get<-1>(iOps))::Operation::WriteDataType;
                                using DisabledTFI = ThreadFusionInfo<ReadIT, WriteOT, false>;
                                execute_instantiable_operations<DisabledTFI>(staticIterator, currentThread, iOps);
                                currentX++;
                            }
                        }
                    }
                }
            }
        }
        FK_HOST_FUSE auto build(const Details& details, const Operations& iOps) {
            return InstantiableDPP<TransformDPP<Details>>{ details, iOps };
        }
    };

    template <>
    struct TransformDPP<void> {
        template <bool THREAD_FUSION, typename FirstIOp, typename... IOps>
        FK_HOST_FUSE auto build(const FirstIOp& firstIOp, const IOps&... iOps) {
            using Details = TransformDPPDetails<THREAD_FUSION, FirstIOp, IOps...>;
            using TFI = typename Details::TFI;
            const Tuple<FirstIOp, IOps...> operations{ firstIOp, iOps... };

            if constexpr (TFI::ENABLED) {
                const ActiveThreads initAT = FirstIOp::getActiveThreads(firstIOp);
                const ActiveThreads gridActiveThreads(static_cast<uint>(ceil(initAT.x / static_cast<float>(TFI::elems_per_thread))),
                                                      initAT.y, initAT.z);
                const bool threadDivisible = isThreadDivisible<TFI::ENABLED>(TFI::elems_per_thread, firstIOp, iOps...);
                const Details details{ gridActiveThreads, threadDivisible };

                return TransformDPP<Details>::build(details, operations);
            } else {
                return TransformDPP<Details>::build(Details{}, operations);
            }
        }

        template <bool THREAD_FUSION, typename... IOps>
        FK_HOST_FUSE auto build_from_tuple(const Tuple<IOps...>& iOps)
            -> decltype(apply(std::declval<TransformDPP<void>::build<THREAD_FUSION, IOps...>>(), std::declval<Tuple<IOps...>>())) {
            return apply(build<THREAD_FUSION, IOps...>, iOps);
        }
    };

    template <typename SequenceSelector, typename ITransformDPPsTuple = void>
    struct DivergentBatchTransformDPP {
        using Operations = ITransformDPPsTuple;
        using InstanceType = DPP_UnaryType;
        template <typename IType>
        static constexpr bool is = std::is_same_v<IType, InstanceType>;
        private:
            template <int OpSequenceNumber, typename ITransformDPP, typename... ITransformDPPs>
            FK_DEVICE_FUSE void divergent_operate(const uint& z, const ITransformDPP& tDPP,
                                                  const ITransformDPPs&... tDPPs) {
                if (OpSequenceNumber == SequenceSelector::at(z)) {
                    ITransformDPP::DPP::exec(tDPP.details, tDPP.operations);
                } else if constexpr (sizeof...(tDPPs) > 0) {
                    divergent_operate<OpSequenceNumber + 1>(z, tDPPs...);
                }
            }
            template <size_t... Idx>
            FK_DEVICE_FUSE void exec_helper(const std::index_sequence<Idx...>&, const Operations& iTDPPs) {
                const cg::thread_block g = cg::this_thread_block();
                const uint z = g.group_index().z;
                divergent_operate<1>(z, get<Idx>(iTDPPs)...);
            }

            template <size_t... Idx>
            FK_HOST_DEVICE_FUSE uint addZThreads(std::index_sequence<Idx...>&, const Operations& operations) {
                return (get_type_t<Idx, Operations>::DPP::getActivethreads(get<Idx>(operations).params, get<Idx>(operations).operations) + ...);
            }

        public:
            FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const Operations& operations) {
                const auto firstITDPP = get<0>(operations);
                using FirstITDPP = decltype(firstITDPP);

                // Asuming all TransformDPPs have the same AT in x and y
                const ActiveThreads onePlane = FirstITDPP::getActiveThreads(firstITDPP.details, firstITDPP.operations);
                // This will be the number of required grid planes to feed all the TransformDPPs
                const uint z = addZThreads(indexSequence<Operations::size>, operations);
                return ActiveThreads(onePlane.x, onePlane.y, z);
            }

            FK_DEVICE_FUSE void exec(const Operations& iTDPPs) {
                exec_helper(std::make_index_sequence<Operations::size>{}, iTDPPs);
            }


    };

    template <typename SequenceSelector>
    struct DivergentBatchTransformDPP<SequenceSelector, void> {
    private:
        template <typename... IOps, typename... IOSeqs>
        FK_HOST_FUSE auto build_helper(const InstantiableOperationSequence<IOps...>& iOps, const IOSeqs&... iOSeqs) {
            const auto iTDPP = apply(TransformDPP<void>::build<false, IOps...>, iOps.instantiableOperations);

            if constexpr (sizeof...(IOSeqs) > 0) {
                return cat(make_tuple(iTDPP), build_helper(iOSeqs...));
            } else {
                return make_tuple(iTDPP);
            }
        }
    public:
        template <typename... IOSeqs>
        FK_HOST_FUSE auto build(const IOSeqs&... iOSeqs) {
            const auto iTDPP_Tuple = build_helper(iOSeqs...);
            return InstantiableDPP<DivergentBatchTransformDPP<SequenceSelector, decltype(iTDPP_Tuple)>>{iTDPP_Tuple};
        }
    };

    template <typename IDPP>
    FK_DEVICE_CNST void executeDPP(const IDPP& iDPP) {
        if constexpr (IDPP::template is<DPP_UnaryType>) {
            IDPP::DPP::exec(iDPP.operations);
        } else {
            IDPP::DPP::exec(iDPP.details, iDPP.operations);
        }
    }

    template <size_t NumDPPs, size_t CurrentDPP, typename IDPP>
    FK_DEVICE_CNST void executeDPPAndSync(const IDPP& iDPP) {
        executeDPP(iDPP);
        if constexpr (CurrentDPP < (NumDPPs - 1)) {
            cg::this_grid().sync();
        }
    }

    template <size_t... Idx, typename... IDPPs>
    FK_DEVICE_CNST void launchDPPs_Kernel_helper(const std::index_sequence<Idx...>&, const IDPPs&... iDPPs) {
        (executeDPPAndSync<sizeof...(Idx), Idx>(iDPPs), ...);
    }

    template <typename... IDPPs>
    __global__ void launchDPPs_Kernel(const __grid_constant__ IDPPs... iDPPs) {
        constexpr auto iterator = indexSequence<sizeof...(IDPPs)>;
        launchDPPs_Kernel_helper(iterator, iDPPs...);
    }
} // namespace fk

#endif
