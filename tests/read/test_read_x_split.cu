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

#include <vector>

#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>

template <int CV_TYPE_I, int CV_TYPE_O, int BATCH>
bool test_read_convert_split(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        try {
            struct Parameters { cv::Scalar init; };

            std::vector<Parameters> params = {
                {{2u}},
                {{2u, 37u}},
                {{2u, 37u, 128u}},
                {{2u, 37u, 128u, 20u}}
            };

            cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;

            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
            cv::cuda::GpuMat d_cvTemp(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_O, val_init);
            std::vector<cv::cuda::GpuMat> d_output_cv(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::cuda::GpuMat> d_output_cvGS(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::Mat> h_cvResults(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::Mat> h_cvGSResults(CV_MAT_CN(CV_TYPE_O));

            for (int i = 0; i < CV_MAT_CN(CV_TYPE_I); i++) {
                d_output_cv.at(i).create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_MAT_DEPTH(CV_TYPE_O));
                h_cvResults.at(i).create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_MAT_DEPTH(CV_TYPE_O));
                d_output_cvGS.at(i).create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_MAT_DEPTH(CV_TYPE_O));
                h_cvGSResults.at(i).create(NUM_ELEMS_Y, NUM_ELEMS_X, CV_MAT_DEPTH(CV_TYPE_O));
            }
            d_input.convertTo(d_cvTemp, CV_TYPE_O, cv_stream);
            cv::cuda::split(d_cvTemp, d_output_cv, cv_stream);
            cvGS::executeOperations(d_input, cv_stream,
                                    cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(),
                                    cvGS::split<CV_TYPE_O>(d_output_cvGS));
            // Verify results
            for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
                d_output_cv.at(i).download(h_cvResults.at(i), cv_stream);
                d_output_cvGS.at(i).download(h_cvGSResults.at(i), cv_stream);
            }

            cv_stream.waitForCompletion();

            cv::Mat diff(NUM_ELEMS_Y, NUM_ELEMS_X, CV_MAT_DEPTH(CV_TYPE_O));
            for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
                diff = cv::abs(h_cvResults.at(i) - h_cvGSResults.at(i));
                passed &= checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
            }
        }
        catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }

        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "test_read_convert_split<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            }
            else {
                std::stringstream ss;
                ss << "test_read_convert_split<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

int launch() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_read_convert_split"] = true;

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_read_convert_split"] &= test_read_convert_split<CV_INPUT, CV_OUTPUT, 1>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

#ifdef ENABLE_BENCHMARK

    // Warming up for the benchmarks
    warmup = true;
    LAUNCH_TESTS(CV_8UC2, CV_32FC2)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_8SC2, CV_32FC2)
    LAUNCH_TESTS(CV_8SC3, CV_32FC3)
    LAUNCH_TESTS(CV_8SC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC2, CV_32FC2)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC2, CV_32FC2)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC2, CV_32FC2)
    LAUNCH_TESTS(CV_32SC3, CV_32FC3)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC2, CV_64FC2)
    LAUNCH_TESTS(CV_32FC3, CV_64FC3)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)
    warmup = false;

#endif

    LAUNCH_TESTS(CV_8UC2, CV_32FC2)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_8SC2, CV_32FC2)
    LAUNCH_TESTS(CV_8SC3, CV_32FC3)
    LAUNCH_TESTS(CV_8SC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC2, CV_32FC2)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC2, CV_32FC2)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC2, CV_32FC2)
    LAUNCH_TESTS(CV_32SC3, CV_32FC3)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC2, CV_64FC2)
    LAUNCH_TESTS(CV_32FC3, CV_64FC3)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)

#undef LAUNCH_TESTS

    CLOSE_BENCHMARK

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }

    return returnValue;
}
