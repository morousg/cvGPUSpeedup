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

#include <testUtils.h>
#include <cvGPUSpeedup.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

template <int T>
bool checkResults(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& h_comparison1C) {
    cv::Mat h_comparison(NUM_ELEMS_Y, NUM_ELEMS_X, T);
    cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, T, 0.00001);
    cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_GT);

#ifdef CVGS_DEBUG
    for (int y=0; y<h_comparison1C.rows; y++) {
        for (int x=0; x<h_comparison1C.cols; x++) {
            CUDA_T(T) value = h_comparison1C.at<CUDA_T(T)>(y,x);
            if (value > 0.001) {
                std::cout << value << ", ";
            }
        }
        std::cout << std::endl;
    }
#endif
    
    int errors = cv::countNonZero(h_comparison);
    return errors == 0;
}

template <int T>
bool compareAndCheck(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& cvVersion, cv::Mat& cvGSVersion) {
    bool passed = true;
    cv::Mat diff = cv::abs(cvVersion - cvGSVersion);
    std::vector<cv::Mat> h_comparison1C(CV_MAT_CN(T));
    cv::split(diff, h_comparison1C);

    for (int i=0; i<CV_MAT_CN(T); i++) {
        passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C.at(i));
    }
    return passed;
}

template <int I, int OC>
bool testSplitOutputOperation(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
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

        cv::Scalar val_init = params.at(CV_MAT_CN(OC)-1).init;
        cv::Scalar val_alpha = params.at(CV_MAT_CN(OC)-1).alpha;
        cv::Scalar val_sub = params.at(CV_MAT_CN(OC)-1).val_sub;
        cv::Scalar val_div = params.at(CV_MAT_CN(OC)-1).val_div;

        try {
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);
            cv::cuda::GpuMat d_crop = d_input(cv::Rect2d(cv::Point2d(200, 200), cv::Point2d(260, 320)));
            cv::Size up(64, 128);
            cv::cuda::GpuMat d_up(up, I);
            cv::cuda::GpuMat d_temp(up, OC);
            cv::cuda::GpuMat d_temp2(up, OC);

            cv::Mat diff(up, CV_MAT_DEPTH(OC));
            std::vector<cv::Mat> h_cvResults(CV_MAT_CN(OC));
            std::vector<cv::Mat> h_cvGSResults(CV_MAT_CN(OC));
            std::vector<cv::cuda::GpuMat> d_output_cv(CV_MAT_CN(OC));
            std::vector<cv::cuda::GpuMat> d_output_cvGS(CV_MAT_CN(OC));

            for (int i=0; i<CV_MAT_CN(I); i++) {
                d_output_cv.at(i).create(up, CV_MAT_DEPTH(OC));
                h_cvResults.at(i).create(up, CV_MAT_DEPTH(OC));
                d_output_cvGS.at(i).create(up, CV_MAT_DEPTH(OC));
                h_cvGSResults.at(i).create(up, CV_MAT_DEPTH(OC));
            }

            // OpenCV version
            cv::cuda::resize(d_crop, d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
            d_up.convertTo(d_temp, OC, alpha, cv_stream);
            cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
            cv::cuda::divide(d_temp2, val_div, d_temp, 1.0, -1, cv_stream);
            cv::cuda::split(d_temp, d_output_cv, cv_stream);

            // cvGPUSpeedup version
            cvGS::executeOperations<I>(cv_stream,
                                       cvGS::resize<I, cv::INTER_LINEAR>(d_input, up, 0., 0.),
                                       cvGS::convertTo<I, OC>(),
                                       cvGS::multiply<OC>(val_alpha),
                                       cvGS::subtract<OC>(val_sub),
                                       cvGS::divide<OC>(val_div),
                                       cvGS::split<OC>(d_output_cvGS));

            // Looking at Nsight Systems, with an RTX A2000 12GB
            // Speedups are up to 7x, depending on the data type

            // Verify results
            for (int i=0; i<CV_MAT_CN(OC); i++) {
                d_output_cv.at(i).download(h_cvResults.at(i), cv_stream);
                d_output_cvGS.at(i).download(h_cvGSResults.at(i), cv_stream);
            }

            cv_stream.waitForCompletion();

            for (int i=0; i<CV_MAT_CN(OC); i++) {
                diff = cv::abs(h_cvResults.at(i) - h_cvGSResults.at(i));
                passed &= checkResults<CV_MAT_DEPTH(OC)>(diff.cols, diff.rows, diff);
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
                ss << "testNoDefinedOutputOperation<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "testNoDefinedOutputOperation<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

template <int I, int OC>
bool testNoDefinedOutputOperation(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {

        struct Parameters {
            cv::Scalar init;
            cv::Scalar val_sub;
            cv::Scalar val_mul;
            cv::Scalar val_div;
        };

        std::vector<Parameters> params = {
            {{2u}, {0.3f}, {1.f}, {3.2f}},
            {{2u, 37u}, {0.3f, 0.3f}, {1.f, 4.f}, {3.2f, 0.6f}},
            {{2u, 37u, 128u}, {0.3f, 0.3f, 0.3f}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f}},
            {{2u, 37u, 128u, 20u}, {0.3f, 0.3f, 0.3f, 0.3f}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}
        };

        cv::Scalar val_init = params.at(CV_MAT_CN(OC)-1).init;
        cv::Scalar val_sub = params.at(CV_MAT_CN(OC)-1).val_sub;
        cv::Scalar val_mul = params.at(CV_MAT_CN(OC)-1).val_mul;
        cv::Scalar val_div = params.at(CV_MAT_CN(OC)-1).val_div;
        cv::Scalar val_add = params.at(CV_MAT_CN(OC)-1).val_div;

        try {
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);
            cv::cuda::GpuMat d_temp(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::cuda::GpuMat d_output_cv(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, OC);

            cv::Mat h_cvResults(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, OC);

            // OpenCV version
            d_input.convertTo(d_temp, OC, cv_stream);
            cv::cuda::subtract(d_temp, val_sub, d_output_cv, cv::noArray(), -1, cv_stream);
            cv::cuda::multiply(d_output_cv, val_mul, d_temp, 1.0, -1, cv_stream);
            cv::cuda::divide(d_temp, val_div, d_output_cv, 1.0, -1, cv_stream);
            cv::cuda::add(d_output_cv, val_add, d_output_cv, cv::noArray(), -1, cv_stream);     

            // cvGPUSpeedup version
            cvGS::executeOperations<I, OC>(d_input, d_output_cvGS, cv_stream, 
                                            cvGS::convertTo<I, OC>(),
                                            cvGS::subtract<OC>(val_sub),
                                            cvGS::multiply<OC>(val_mul),
                                            cvGS::divide<OC>(val_div),
                                            cvGS::add<OC>(val_add));

            // Looking at Nsight Systems, with an RTX A2000 12GB
            // Speedups are up to 7x, depending on the data type

            // Verify results
            d_output_cv.download(h_cvResults, cv_stream);
            d_output_cvGS.download(h_cvGSResults, cv_stream);

            cv_stream.waitForCompletion();

            passed = compareAndCheck<OC>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvResults, h_cvGSResults);
            
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }

        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "testNoDefinedOutputOperation<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "testNoDefinedOutputOperation<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }

    }

    return passed;
}

template <int I, int O>
bool testResize(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {

        struct Parameters {
            cv::Scalar init;
        };

        std::vector<Parameters> params = {
            {{2u}},
            {{2u, 37u}},
            {{2u, 37u, 128u}},
            {{2u, 37u, 128u, 20u}}
        };

        cv::Scalar val_init = params.at(CV_MAT_CN(I)-1).init;

        try {

            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);

            cv::Size up(3870, 2260); // x,y
            cv::Size down(300, 500); // x,y

            cv::cuda::GpuMat d_down(down, I);
            cv::cuda::GpuMat d_up(up, I);

            cv::cuda::GpuMat d_down_cvGS(down, I);
            cv::cuda::GpuMat d_up_cvGS(up, I);

            cv::cuda::resize(d_input, d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
            cv::cuda::resize(d_input, d_down, down, 0., 0., cv::INTER_LINEAR, cv_stream);

            cvGS::executeOperations<I>(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, up, 0., 0.), cvGS::write<I>(d_up_cvGS));
            cvGS::executeOperations<I>(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, down, 0., 0.), cvGS::write<I>(d_down_cvGS));
            cv::Mat h_up, h_up_cvGS;
            cv::Mat h_down, h_down_cvGS;

            d_up.download(h_up, cv_stream);
            d_up_cvGS.download(h_up_cvGS, cv_stream);
            d_down.download(h_down, cv_stream);
            d_down_cvGS.download(h_down_cvGS, cv_stream);

            cv_stream.waitForCompletion();

            passed &= compareAndCheck<I>(up.width, up.height, h_up, h_up_cvGS);
            passed &= compareAndCheck<I>(down.width, down.height, h_down, h_down_cvGS);

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
                ss << "testResize<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "testResize<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
results["testNoDefinedOutputOperation"] &= testNoDefinedOutputOperation<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
results["testSplitOutputOperation"] &= testSplitOutputOperation<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
results["testResize"] &= testResize<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

#define LAUNCH_TESTS_NO_SPLIT(CV_INPUT, CV_OUTPUT) \
results["testNoDefinedOutputOperation"] &= testNoDefinedOutputOperation<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
results["testResize"] &= testResize<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

int main() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["testNoDefinedOutputOperation"] = true;
    results["testSplitOutputOperation"] = true;
    results["testResize"] = true;

    LAUNCH_TESTS_NO_SPLIT(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS_NO_SPLIT(CV_8SC1, CV_32FC1)
    LAUNCH_TESTS_NO_SPLIT(CV_16UC1, CV_32FC1)
    LAUNCH_TESTS_NO_SPLIT(CV_16SC1, CV_32FC1)
    LAUNCH_TESTS_NO_SPLIT(CV_32SC1, CV_32FC1)
    LAUNCH_TESTS_NO_SPLIT(CV_32FC1, CV_32FC1)
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

    #undef LAUNCH_TESTS_NO_SPLIT
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