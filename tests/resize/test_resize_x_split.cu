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

template <int CV_TYPE_I, int CV_TYPE_O>
bool test_resize_split_one(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        struct Parameters {
            cv::Scalar init;
            cv::Scalar alpha;
            cv::Scalar val_sub;
            cv::Scalar val_div;
        };

        double alpha = 0.3;

        std::vector<Parameters> params = {
            {{2u}, {alpha}, {1.f}, {3.2f}},
            {{2u, 37u}, {alpha, alpha}, {1.f, 4.f}, {3.2f, 0.6f}},
            {{2u, 37u, 128u}, {alpha, alpha, alpha}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f}},
            {{2u, 37u, 128u, 20u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}
        };

        cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O)-1).init;
        cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O)-1).alpha;
        cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O)-1).val_sub;
        cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O)-1).val_div;

        try {
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
            cv::Rect2d crop(cv::Point2d(200, 200), cv::Point2d(260, 320));

            cv::Size up(64, 128);
            cv::cuda::GpuMat d_up(up, CV_TYPE_I);
            cv::cuda::GpuMat d_temp(up, CV_TYPE_O);
            cv::cuda::GpuMat d_temp2(up, CV_TYPE_O);

            cv::Mat diff(up, CV_MAT_DEPTH(CV_TYPE_O));
            std::vector<cv::Mat> h_cvResults(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::Mat> h_cvGSResults(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::cuda::GpuMat> d_output_cv(CV_MAT_CN(CV_TYPE_O));
            std::vector<cv::cuda::GpuMat> d_output_cvGS(CV_MAT_CN(CV_TYPE_O));

            for (int i=0; i<CV_MAT_CN(CV_TYPE_I); i++) {
                d_output_cv.at(i).create(up, CV_MAT_DEPTH(CV_TYPE_O));
                h_cvResults.at(i).create(up, CV_MAT_DEPTH(CV_TYPE_O));
                d_output_cvGS.at(i).create(up, CV_MAT_DEPTH(CV_TYPE_O));
                h_cvGSResults.at(i).create(up, CV_MAT_DEPTH(CV_TYPE_O));
            }

            // OpenCV version
            cv::cuda::resize(d_input(crop), d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
            d_up.convertTo(d_temp, CV_TYPE_O, alpha, cv_stream);
            cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
            cv::cuda::divide(d_temp2, val_div, d_temp, 1.0, -1, cv_stream);
            cv::cuda::split(d_temp, d_output_cv, cv_stream);
            
            // cvGPUSpeedup version
            cvGS::executeOperations(cv_stream,
                                    cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR>(d_input(crop), up, 0., 0.),
                                    cvGS::multiply<CV_TYPE_O>(val_alpha),
                                    cvGS::subtract<CV_TYPE_O>(val_sub),
                                    cvGS::divide<CV_TYPE_O>(val_div),
                                    cvGS::split<CV_TYPE_O>(d_output_cvGS));

            // Verify results
            for (int i=0; i<CV_MAT_CN(CV_TYPE_O); i++) {
                d_output_cv.at(i).download(h_cvResults.at(i), cv_stream);
                d_output_cvGS.at(i).download(h_cvGSResults.at(i), cv_stream);
            }

            cv_stream.waitForCompletion();

            for (int i=0; i<CV_MAT_CN(CV_TYPE_O); i++) {
                diff = cv::abs(h_cvResults.at(i) - h_cvGSResults.at(i));
                passed &= checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
            }

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
                ss << "test_resize_split_one<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_split_one<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
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
    results["test_resize_split_one"] = true;
    

    #define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_resize_split_one"] = test_resize_split_one<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)

    #undef LAUNCH_TESTS

    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
        }
    }

    return 0;
}