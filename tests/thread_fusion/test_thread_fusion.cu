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

#include "tests/main.h"

#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>

#include <opencv2/cudaimgproc.hpp>

#include <type_traits>

template <typename OriginalType>
bool testThreadFusion() {
    constexpr OriginalType eightNumbers[8]{ static_cast<OriginalType>(10),
                                            static_cast<OriginalType>(2),
                                            static_cast<OriginalType>(3),
                                            static_cast<OriginalType>(4),
                                            static_cast<OriginalType>(10),
                                            static_cast<OriginalType>(2),
                                            static_cast<OriginalType>(3),
                                            static_cast<OriginalType>(4)};

    using BTInfo = fk::ThreadFusionInfo<OriginalType, true>;

    const typename BTInfo::BiggerReadType biggerType = ((typename BTInfo::BiggerReadType*) eightNumbers)[0];

    if constexpr (BTInfo::elems_per_thread == 1) {
        return eightNumbers[0] == biggerType;
    } else if constexpr (BTInfo::elems_per_thread == 2) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        return (data0 == eightNumbers[0]) && (data1 == eightNumbers[1]);
    } else if constexpr (BTInfo::elems_per_thread == 4) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        const OriginalType data2 = BTInfo::template get<2>(biggerType);
        const OriginalType data3 = BTInfo::template get<3>(biggerType);
        return data0 == eightNumbers[0] && data1 == eightNumbers[1] &&
               data2 == eightNumbers[2] && data3 == eightNumbers[3];
    } else if constexpr (BTInfo::elems_per_thread == 8) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        const OriginalType data2 = BTInfo::template get<2>(biggerType);
        const OriginalType data3 = BTInfo::template get<3>(biggerType);
        const OriginalType data4 = BTInfo::template get<4>(biggerType);
        const OriginalType data5 = BTInfo::template get<5>(biggerType);
        const OriginalType data6 = BTInfo::template get<6>(biggerType);
        const OriginalType data7 = BTInfo::template get<7>(biggerType);
        return data0 == eightNumbers[0] && data1 == eightNumbers[1] &&
               data2 == eightNumbers[2] && data3 == eightNumbers[3] &&
               data4 == eightNumbers[4] && data5 == eightNumbers[5] &&
               data6 == eightNumbers[6] && data7 == eightNumbers[7];
    }
}

namespace fk {
    template <typename OriginalType>
    bool testThreadFusionAggregate() {
        constexpr OriginalType fourNumbers[4]{ fk::make_<OriginalType>(10),
                                               fk::make_<OriginalType>(2),
                                               fk::make_<OriginalType>(3),
                                               fk::make_<OriginalType>(4) };

        using BTInfo = fk::ThreadFusionInfo<OriginalType, true>;

        const typename BTInfo::BiggerReadType biggerType = ((typename BTInfo::BiggerReadType*) fourNumbers)[0];

        using Reduction = VectorReduce<VectorType_t<uchar, (cn<OriginalType>)>, Sum<uchar>>;

        if constexpr (BTInfo::elems_per_thread == 1) {
            return Reduction::exec(biggerType == fourNumbers[0]);
        } else if constexpr (BTInfo::elems_per_thread == 2) {
            const OriginalType data0 = BTInfo::template get<0>(biggerType);
            const OriginalType data1 = BTInfo::template get<1>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                   Reduction::exec(data1 == fourNumbers[1]);
        } else if constexpr (BTInfo::elems_per_thread == 4) {
            const OriginalType data0 = BTInfo::template get<0>(biggerType);
            const OriginalType data1 = BTInfo::template get<1>(biggerType);
            const OriginalType data2 = BTInfo::template get<2>(biggerType);
            const OriginalType data3 = BTInfo::template get<3>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                   Reduction::exec(data1 == fourNumbers[1]) &&
                   Reduction::exec(data2 == fourNumbers[2]) &&
                   Reduction::exec(data3 == fourNumbers[3]);
        }
    }
}

template <int I>
bool testThreadFusionTimes(uint NUM_ELEMS_X, uint NUM_ELEMS_Y, cv::cuda::Stream& cv_stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    struct Parameters {
        cv::Scalar init;
    };

    std::vector<Parameters> params = { {{2u}}, {{2u, 37u}}, {{2u, 37u, 128u}}, {{2u, 37u, 128u, 20u}} };

    cv::Scalar val_init = params.at(CV_MAT_CN(I) - 1).init;

    try {
        cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);
        cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, I);
        cv::cuda::GpuMat d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, I);

        cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, I);
        cv::Mat h_cvGSResults_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, I);

        // cvGPUSpeedup non fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(I)>>
            read{{cvGS::gpuMat2RawPtr2D<CUDA_T(I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1}};
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(I)>>
            write{cvGS::gpuMat2RawPtr2D<CUDA_T(I)>(d_output_cvGS)};
        cvGS::executeOperations(cv_stream, read, write);

        // cvGPUSpeedup fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(I), fk::ThreadFusionInfo<CUDA_T(I), true>>>
            readTF{{cvGS::gpuMat2RawPtr2D<CUDA_T(I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1 }};
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(I), true>>
            writeTF{cvGS::gpuMat2RawPtr2D<CUDA_T(I)>(d_output_cvGS_ThreadFusion)};
        cvGS::executeOperations(cv_stream, readTF, writeTF);

        // Verify results
        d_output_cvGS_ThreadFusion.download(h_cvGSResults_ThreadFusion, cv_stream);
        d_output_cvGS.download(h_cvGSResults, cv_stream);

        cv_stream.waitForCompletion();

        passed = compareAndCheck<I>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvGSResults_ThreadFusion, h_cvGSResults);
    }
    catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<I>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        }
        else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<I>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

int launch() {
    bool passed = true;

    passed &= testThreadFusion<uchar>();
    passed &= testThreadFusion<char>();
    passed &= testThreadFusion<ushort>();
    passed &= testThreadFusion<short>();
    passed &= testThreadFusion<uint>();
    passed &= testThreadFusion<int>();
    passed &= testThreadFusion<ulong>();
    passed &= testThreadFusion<long>();
    passed &= testThreadFusion<ulonglong>();
    passed &= testThreadFusion<longlong>();
    passed &= testThreadFusion<float>();
    passed &= testThreadFusion<double>();

#define LAUNCH_AGGREGATE(type) \
    passed &= fk::testThreadFusionAggregate<type ## 2>(); \
    passed &= fk::testThreadFusionAggregate<type ## 3>(); \
    passed &= fk::testThreadFusionAggregate<type ## 4>();

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

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    constexpr uint NUM_ELEMS_X = 3840 * 2;
    constexpr uint NUM_ELEMS_Y = 2160 * 2;
    cv::cuda::Stream cv_stream;

#define LAUNCH_testThreadFusionTimes(BASE) \
    passed &= testThreadFusionTimes<BASE ## C1>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream); \
    passed &= testThreadFusionTimes<BASE ## C2>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream); \
    passed &= testThreadFusionTimes<BASE ## C3>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream); \
    passed &= testThreadFusionTimes<BASE ## C4>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream);

    LAUNCH_testThreadFusionTimes(CV_8U)
    LAUNCH_testThreadFusionTimes(CV_8S)
    LAUNCH_testThreadFusionTimes(CV_16U)
    LAUNCH_testThreadFusionTimes(CV_16S)
    LAUNCH_testThreadFusionTimes(CV_32S)
    LAUNCH_testThreadFusionTimes(CV_32F)
    LAUNCH_testThreadFusionTimes(CV_64F)
#undef LAUNCH_testThreadFusionTimes

    if (passed) {
        std::cout << "test_thread_fusion Passed!!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_thread_fusion Failed!!!" << std::endl;
        return -1;
    }
}
