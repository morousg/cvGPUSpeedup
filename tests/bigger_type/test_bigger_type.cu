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

#include <fused_kernel/core/execution_model/bigger_type.cuh>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>

#include "tests/main.h"

#include <type_traits>

template <typename OriginalType>
bool testBiggerType() {
    constexpr OriginalType fourNumbers[4]{ static_cast<OriginalType>(10),
                                           static_cast<OriginalType>(2),
                                           static_cast<OriginalType>(3),
                                           static_cast<OriginalType>(4) };

    using NewType = fk::BiggerType_t<OriginalType>;

    const typename NewType::type biggerType = ((typename NewType::type*) fourNumbers)[0];

    if constexpr (NewType::times_bigger == 1) {
        return fourNumbers[0] == biggerType;
    } else if constexpr (NewType::times_bigger == 2) {
        const OriginalType data0 = NewType::template get<0>(biggerType);
        const OriginalType data1 = NewType::template get<1>(biggerType);
        return (data0 == fourNumbers[0]) && (data1 == fourNumbers[1]);
    } else if constexpr (NewType::times_bigger == 4) {
        const OriginalType data0 = NewType::template get<0>(biggerType);
        const OriginalType data1 = NewType::template get<1>(biggerType);
        const OriginalType data2 = NewType::template get<2>(biggerType);
        const OriginalType data3 = NewType::template get<3>(biggerType);
        return data0 == fourNumbers[0] && data1 == fourNumbers[1] &&
               data2 == fourNumbers[2] && data3 == fourNumbers[3];
    }
}

namespace fk {
    template <typename OriginalType>
    bool testBiggerTypeAggregate() {
        constexpr OriginalType fourNumbers[4]{ fk::make_<OriginalType>(10),
                                               fk::make_<OriginalType>(2),
                                               fk::make_<OriginalType>(3),
                                               fk::make_<OriginalType>(4) };

        using NewType = fk::BiggerType_t<OriginalType>;

        const typename NewType::type biggerType = ((typename NewType::type*) fourNumbers)[0];

        using Reduction = VectorReduce<VectorType_t<uchar, (cn<OriginalType>)>, Sum<uchar>>;

        if constexpr (NewType::times_bigger == 1) {
            return Reduction::exec(biggerType == fourNumbers[0]);
        } else if constexpr (NewType::times_bigger == 2) {
            const OriginalType data0 = NewType::template get<0>(biggerType);
            const OriginalType data1 = NewType::template get<1>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                   Reduction::exec(data1 == fourNumbers[1]);
        } else if constexpr (NewType::times_bigger == 4) {
            const OriginalType data0 = NewType::template get<0>(biggerType);
            const OriginalType data1 = NewType::template get<1>(biggerType);
            const OriginalType data2 = NewType::template get<2>(biggerType);
            const OriginalType data3 = NewType::template get<3>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                   Reduction::exec(data1 == fourNumbers[1]) &&
                   Reduction::exec(data2 == fourNumbers[2]) &&
                   Reduction::exec(data3 == fourNumbers[3]);
        }
    }
}
int launch() {
    bool passed = true;

    passed &= testBiggerType<uchar>();
    passed &= testBiggerType<char>();
    passed &= testBiggerType<short>();
    passed &= testBiggerType<ushort>();
    passed &= testBiggerType<int>();
    passed &= testBiggerType<uint>();
    passed &= testBiggerType<long>();
    passed &= testBiggerType<ulong>();
    passed &= testBiggerType<longlong>();
    passed &= testBiggerType<ulonglong>();
    passed &= testBiggerType<float>();
    passed &= testBiggerType<double>();

#define LAUNCH_AGGREGATE(type) \
    passed &= fk::testBiggerTypeAggregate<type ## 2>(); \
    passed &= fk::testBiggerTypeAggregate<type ## 3>(); \
    passed &= fk::testBiggerTypeAggregate<type ## 4>();

    LAUNCH_AGGREGATE(char)
    LAUNCH_AGGREGATE(uchar)
    LAUNCH_AGGREGATE(short)
    LAUNCH_AGGREGATE(ushort)
    LAUNCH_AGGREGATE(int)
    LAUNCH_AGGREGATE(uint)
    LAUNCH_AGGREGATE(long)
    LAUNCH_AGGREGATE(ulong)
    LAUNCH_AGGREGATE(longlong)
    LAUNCH_AGGREGATE(ulonglong)
    LAUNCH_AGGREGATE(float)
    LAUNCH_AGGREGATE(double)

#undef LAUNCH_AGGREGATE

    if (passed) {
        std::cout << "test_bigger_type Passed!!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_bigger_type Failed!!!" << std::endl;
        return -1;
    }
}
