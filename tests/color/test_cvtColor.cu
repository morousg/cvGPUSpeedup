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

#include <sstream>

#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>
#include "tests/nvtx.h"

template <int CV_TYPE_I, int CV_TYPE_O, cv::ColorConversionCodes CC>
bool test_cvtColor(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        struct Parameters {
            const cv::Scalar init;
        };

        const Parameters one{ {1u} };
        const Parameters two{ {1u, 2u} };
        const Parameters three{ {10u, 100u, 200u} };
        const Parameters four{ {1u, 2u, 3u, 4u} };
        const std::array<Parameters, 4> params{ one, two, three, four };

        const cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_I) - 1).init;
        try {
            cv::cuda::GpuMat d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
            cv::cuda::GpuMat d_output_cv((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_O);
            cv::cuda::GpuMat d_output_cvGS((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_O);
            cv::Mat h_cvResults;
            cv::Mat h_cvGSResults;

            cv::Mat diff((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_O);
            // OpenCV version
            PUSH_RANGE("Launching OpenCV")
            cv::cuda::cvtColor(d_input, d_output_cv, CC, 0, cv_stream);
            POP_RANGE
            // cvGPUSpeedup
            PUSH_RANGE("Launching cvGS")
            cvGS::executeOperations(d_input, d_output_cvGS, cv_stream, cvGS::cvtColor<CC, CV_TYPE_I, CV_TYPE_O>());
            POP_RANGE

            d_output_cv.download(h_cvResults, cv_stream);
            d_output_cvGS.download(h_cvGSResults, cv_stream);
            cv_stream.waitForCompletion();

            // Verify results
            passed = compareAndCheck<CV_TYPE_O>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvResults, h_cvGSResults);
        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            }
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }
        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "test_cvtColor<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_cvtColor<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

int main() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_cvtColor"] = true;

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT, CC) \
    results["test_cvtColor"] &= test_cvtColor<CV_INPUT, CV_OUTPUT, CC>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

    LAUNCH_TESTS(CV_8UC3, CV_8UC3, cv::COLOR_RGB2BGR)
    LAUNCH_TESTS(CV_8UC4, CV_8UC4, cv::COLOR_RGBA2BGRA)
    LAUNCH_TESTS(CV_16UC3, CV_16UC3, cv::COLOR_RGB2BGR)
    LAUNCH_TESTS(CV_16UC4, CV_16UC4, cv::COLOR_RGBA2BGRA)

    LAUNCH_TESTS(CV_8UC3, CV_8UC3, cv::COLOR_BGR2RGB)
    LAUNCH_TESTS(CV_8UC4, CV_8UC4, cv::COLOR_BGRA2RGBA)
    LAUNCH_TESTS(CV_16UC3, CV_16UC3, cv::COLOR_BGR2RGB)
    LAUNCH_TESTS(CV_16UC4, CV_16UC4, cv::COLOR_BGRA2RGBA)

    LAUNCH_TESTS(CV_8UC3, CV_8UC1, cv::COLOR_RGB2GRAY)
    LAUNCH_TESTS(CV_8UC4, CV_8UC1, cv::COLOR_RGBA2GRAY)
    LAUNCH_TESTS(CV_16UC3, CV_16UC1, cv::COLOR_RGB2GRAY)
    LAUNCH_TESTS(CV_16UC4, CV_16UC1, cv::COLOR_RGBA2GRAY)

    LAUNCH_TESTS(CV_8UC3, CV_8UC1, cv::COLOR_BGR2GRAY)
    LAUNCH_TESTS(CV_8UC4, CV_8UC1, cv::COLOR_BGRA2GRAY)
    LAUNCH_TESTS(CV_16UC3, CV_16UC1, cv::COLOR_BGR2GRAY)
    LAUNCH_TESTS(CV_16UC4, CV_16UC1, cv::COLOR_BGRA2GRAY)
#undef LAUNCH_TESTS

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
