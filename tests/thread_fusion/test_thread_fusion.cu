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

#include "tests/main.h"

#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include "tests/testsCommon.cuh"
#include "tests/nvtx.h"
#include <cvGPUSpeedup.cuh>

#include <opencv2/cudaimgproc.hpp>

#include <type_traits>

constexpr size_t NUM_EXPERIMENTS = 5;
#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[]{ "Pixels per side" };
#endif
constexpr size_t FIRST_VALUE = 1024;
constexpr size_t INCREMENT = 1024;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

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

    using BTInfo = fk::ThreadFusionInfo<OriginalType, OriginalType, true>;

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

        using BTInfo = fk::ThreadFusionInfo<OriginalType, OriginalType, true>;

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

template <int I, size_t RESOLUTION>
bool testThreadFusionSameTypeIO(cv::cuda::Stream& cv_stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;
    constexpr int CV_TYPE_I = I;
#ifdef ENABLE_BENCHMARK
    constexpr int CV_TYPE_O = I;
    constexpr size_t BATCH = RESOLUTION;
#endif
    constexpr uint NUM_ELEMS_X = (uint)RESOLUTION;
    constexpr uint NUM_ELEMS_Y = (uint)RESOLUTION;

    struct Parameters {
        cv::Scalar init;
    };

    std::vector<Parameters> params = { {{2u}}, {{2u, 37u}}, {{2u, 37u, 128u}}, {{2u, 37u, 128u, 20u}} };

    cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_I) - 1).init;

    try {
        cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
        cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I);
        cv::cuda::GpuMat d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I);

        cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I);
        cv::Mat h_cvGSResults_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I);

        // In this case it's not OpenCV, it's cvGPUSpeedup without thread fusion
        START_OCV_BENCHMARK 
        // cvGPUSpeedup non fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            read{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1} };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_I)>>
            write{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_output_cvGS) };
        cvGS::executeOperations<false>(cv_stream, read, write);

        STOP_OCV_START_CVGS_BENCHMARK
        // cvGPUSpeedup fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            readTF{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1 } };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_I)>>
            writeTF{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_output_cvGS_ThreadFusion) };
        cvGS::executeOperations<true>(cv_stream, readTF, writeTF);

        STOP_CVGS_BENCHMARK

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

template <int I, int O, size_t RESOLUTION>
bool testThreadFusionDifferentTypeIO(cv::cuda::Stream& cv_stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;
    constexpr int CV_TYPE_I = I;
    constexpr int CV_TYPE_O = O;
#ifdef ENABLE_BENCHMARK
    constexpr size_t BATCH = RESOLUTION;
#endif
    constexpr uint NUM_ELEMS_X = (uint)RESOLUTION;
    constexpr uint NUM_ELEMS_Y = (uint)RESOLUTION;

    struct Parameters {
        cv::Scalar init;
    };

    std::vector<Parameters> params = { {{2u}}, {{2u, 37u}}, {{2u, 37u, 128u}}, {{2u, 37u, 128u, 20u}} };

    cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_I) - 1).init;

    try {
        cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
        cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);
        cv::cuda::GpuMat d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);

        cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);
        cv::Mat h_cvGSResults_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);

        // In this case it's not OpenCV, it's cvGPUSpeedup without thread fusion
        START_OCV_BENCHMARK
        // cvGPUSpeedup non fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            read{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1} };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_O)>>
            write{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_O)>(d_output_cvGS) };
        cvGS::executeOperations<false>(cv_stream, read, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), write);

        STOP_OCV_START_CVGS_BENCHMARK
        // cvGPUSpeedup fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            readTF{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1 } };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_O)>>
            writeTF{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_O)>(d_output_cvGS_ThreadFusion) };
        cvGS::executeOperations<true>(cv_stream, readTF, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), writeTF);

        STOP_CVGS_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(h_cvGSResults_ThreadFusion, cv_stream);
        d_output_cvGS.download(h_cvGSResults, cv_stream);

        cv_stream.waitForCompletion();

        passed = compareAndCheck<O>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvGSResults_ThreadFusion, h_cvGSResults);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <int I, int T, int O, cv::ColorConversionCodes CODE, size_t RESOLUTION>
bool testThreadFusionDifferentTypeAndChannelIO(cv::cuda::Stream& cv_stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;
    constexpr int CV_TYPE_I = I;
    constexpr int CV_TYPE_O = O;
#ifdef ENABLE_BENCHMARK
    constexpr size_t BATCH = RESOLUTION;
#endif
    constexpr uint NUM_ELEMS_X = (uint)RESOLUTION;
    constexpr uint NUM_ELEMS_Y = (uint)RESOLUTION;

    struct Parameters {
        cv::Scalar init;
    };

    std::vector<Parameters> params = { {{2u}}, {{2u, 37u}}, {{2u, 37u, 128u}}, {{2u, 37u, 128u, 20u}} };

    cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_I) - 1).init;

    try {
        cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
        cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);
        cv::cuda::GpuMat d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);

        cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);
        cv::Mat h_cvGSResults_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O);

        // In this case it's not OpenCV, it's cvGPUSpeedup without thread fusion
        START_OCV_BENCHMARK
        // cvGPUSpeedup non fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            read{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1} };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_O)>>
            write{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_O)>(d_output_cvGS) };
        cvGS::executeOperations<false>(cv_stream, read, cvGS::convertTo<CV_TYPE_I, T>(), cvGS::cvtColor<CODE, T, CV_TYPE_O>(), write);

        STOP_OCV_START_CVGS_BENCHMARK
        // cvGPUSpeedup fusion version
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(CV_TYPE_I)>>
            readTF{ {cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_I)>(d_input)}, { NUM_ELEMS_X, NUM_ELEMS_Y, 1 } };
        const fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(CV_TYPE_O)>>
            writeTF{ cvGS::gpuMat2RawPtr2D<CUDA_T(CV_TYPE_O)>(d_output_cvGS_ThreadFusion) };
        cvGS::executeOperations<true>(cv_stream, readTF, cvGS::convertTo<CV_TYPE_I, T>(), cvGS::cvtColor<CODE, T, CV_TYPE_O>(), writeTF);

        STOP_CVGS_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(h_cvGSResults_ThreadFusion, cv_stream);
        d_output_cvGS.download(h_cvGSResults, cv_stream);

        cv_stream.waitForCompletion();

        passed = compareAndCheck<CV_TYPE_O>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvGSResults_ThreadFusion, h_cvGSResults);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <size_t... IDX>
bool testThreadFusionSameTypeIO_launcher_impl(cv::cuda::Stream& cv_stream, std::integer_sequence<size_t, IDX...>) {
    bool passed = true;

#define LAUNCH_testThreadFusionSameTypeIO(BASE) \
    passed &= (testThreadFusionSameTypeIO<BASE ## C1, batchValues[IDX]>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C1, batchValues[IDX] + 1>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C2, batchValues[IDX]>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C2, batchValues[IDX] + 1>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C3, batchValues[IDX]>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C3, batchValues[IDX] + 1>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C4, batchValues[IDX]>(cv_stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## C4, batchValues[IDX] + 1>(cv_stream) && ...);

    LAUNCH_testThreadFusionSameTypeIO(CV_8U)
    LAUNCH_testThreadFusionSameTypeIO(CV_8S)
    LAUNCH_testThreadFusionSameTypeIO(CV_16U)
    LAUNCH_testThreadFusionSameTypeIO(CV_16S)
    LAUNCH_testThreadFusionSameTypeIO(CV_32S)
    LAUNCH_testThreadFusionSameTypeIO(CV_32F)
    LAUNCH_testThreadFusionSameTypeIO(CV_64F)
#undef LAUNCH_testThreadFusionTimes

    return passed;
}

bool testThreadFusionSameTypeIO_launcher(cv::cuda::Stream& cv_stream) {
    return testThreadFusionSameTypeIO_launcher_impl(cv_stream, std::make_integer_sequence<size_t, batchValues.size()>());
}

template <size_t... IDX>
bool testThreadFusionDifferentTypeIO_launcher_impl(cv::cuda::Stream& cv_stream, std::integer_sequence<size_t, IDX...>) {
    bool passed = true;

    passed &= (testThreadFusionDifferentTypeIO<CV_8UC1, CV_32FC1, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC1, CV_32FC1, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC2, CV_32FC2, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC2, CV_32FC2, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC3, CV_32FC3, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC3, CV_32FC3, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC4, CV_32FC4, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_8UC4, CV_32FC4, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC1, CV_32FC1, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC1, CV_32FC1, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC2, CV_32FC2, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC2, CV_32FC2, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC3, CV_32FC3, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC3, CV_32FC3, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC4, CV_32FC4, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<CV_16UC4, CV_32FC4, batchValues[IDX] + 1>(cv_stream) && ...);

    return passed;
}

bool testThreadFusionDifferentTypeIO_launcher(cv::cuda::Stream& cv_stream) {
    return testThreadFusionDifferentTypeIO_launcher_impl(cv_stream, std::make_integer_sequence<size_t, batchValues.size()>());
}

template <size_t... IDX>
bool testThreadFusionDifferentTypeAndChannelIO_launcher_impl(cv::cuda::Stream& cv_stream, std::integer_sequence<size_t, IDX...>) {
    bool passed = true;

    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_8UC3, CV_32FC3, CV_32FC4, cv::ColorConversionCodes::COLOR_RGB2RGBA, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_8UC3, CV_32FC3, CV_32FC4, cv::ColorConversionCodes::COLOR_RGB2RGBA, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_8UC4, CV_32FC4, CV_32FC3, cv::ColorConversionCodes::COLOR_RGBA2RGB, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_8UC4, CV_32FC4, CV_32FC3, cv::ColorConversionCodes::COLOR_RGBA2RGB, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC3, CV_8UC3, CV_8UC4, cv::ColorConversionCodes::COLOR_RGB2RGBA, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC3, CV_8UC3, CV_8UC4, cv::ColorConversionCodes::COLOR_RGB2RGBA, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC4, CV_8UC4, CV_8UC3, cv::ColorConversionCodes::COLOR_RGBA2RGB, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC4, CV_8UC4, CV_8UC3, cv::ColorConversionCodes::COLOR_RGBA2RGB, batchValues[IDX] + 1>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC4, CV_8UC4, CV_8UC1, cv::ColorConversionCodes::COLOR_RGBA2GRAY, batchValues[IDX]>(cv_stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<CV_32FC4, CV_8UC4, CV_8UC1, cv::ColorConversionCodes::COLOR_RGBA2GRAY, batchValues[IDX] + 1>(cv_stream) && ...);

    return passed;
}

bool testThreadFusionDifferentTypeAndChannelIO_launcher(cv::cuda::Stream& cv_stream) {
    return testThreadFusionDifferentTypeAndChannelIO_launcher_impl(cv_stream, std::make_integer_sequence<size_t, batchValues.size()>());
}

int launch() {
    bool passed = true;
    {
        PUSH_RANGE_RAII p("testThreadFusion");
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
    }
#define LAUNCH_AGGREGATE(type) \
    passed &= fk::testThreadFusionAggregate<type ## 2>(); \
    passed &= fk::testThreadFusionAggregate<type ## 3>(); \
    passed &= fk::testThreadFusionAggregate<type ## 4>();

    {
        PUSH_RANGE_RAII p("testThreadFusionAggregate");
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
    }
#undef LAUNCH_AGGREGATE

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));
    cv::cuda::Stream cv_stream;

    {
        PUSH_RANGE_RAII p("testThreadFusionSameTypeIO");
        passed &= testThreadFusionSameTypeIO_launcher(cv_stream);
    }
    {
        PUSH_RANGE_RAII p("testThreadFusionDifferentTypeIO");
        passed &= testThreadFusionDifferentTypeIO_launcher(cv_stream);
    }

    {
        PUSH_RANGE_RAII p("testThreadFusionDifferentTypeAndChannelIO");
        passed &= testThreadFusionDifferentTypeAndChannelIO_launcher(cv_stream);
    }

    CLOSE_BENCHMARK

    if (passed) {
        std::cout << "test_thread_fusion Passed!!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_thread_fusion Failed!!!" << std::endl;
        return -1;
    }
}
