/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifdef ENABLE_BENCHMARK

#include "tests/main.h"

#include "tests/testsCommon.cuh"
#include "tests/nvtx.h"

#include <cvGPUSpeedup.cuh>

constexpr char VARIABLE_DIMENSION[]{ "Batch size" };
constexpr size_t NUM_EXPERIMENTS = 1;
constexpr size_t FIRST_VALUE = 50;
constexpr size_t INCREMENT = 1;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <const int BATCH>
bool test_batchresize_aspectratio_x_split3D(cv::cuda::Stream& cv_stream, bool enabled) {
    constexpr int CV_TYPE_I = CV_8UC3;
    constexpr int CV_TYPE_O = CV_32FC3;
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        constexpr double alpha = 0.3;

        const cv::Scalar val_init{ 5u, 5u, 5u };
        const cv::Scalar val_alpha{ alpha, alpha, alpha };
        const cv::Scalar val_sub{ 1.f, 4.f, 3.2f };
        const cv::Scalar val_div{ 3.2f, 0.6f, 11.8f };
        const cv::Scalar val_init_output{ 128.f, 128.f, 128.f };

        try {
            // Setting up OpenCV variables for cvGS
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3, val_init);

            std::array<cv::Rect2d, BATCH> crops_2d;
            constexpr int cropWidth = 30;
            constexpr int cropHeight = 120;
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                crops_2d[crop_i] = cv::Rect2d(cv::Point2d(crop_i, crop_i), cv::Point2d(crop_i + cropWidth, crop_i + cropHeight));
            }

            cv::Size up(64, 128);
            float scaleFactor = up.height / (float)cropHeight;
            int newHeight = up.height;
            int newWidth = static_cast<int>(scaleFactor * cropWidth);
            if (newWidth > up.width) {
                scaleFactor = up.width / (float)cropWidth;
                newWidth = up.width;
                newHeight = static_cast<int> (scaleFactor * cropHeight);
            }
            cv::Size upAspectRatio(newWidth, newHeight);

            cv::cuda::GpuMat d_up(up, CV_8UC3);
            cv::cuda::GpuMat d_temp(up, CV_32FC3);
            cv::cuda::GpuMat d_temp2(up, CV_32FC3);

            std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
            std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

            cv::cuda::GpuMat d_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_32FC3),
                                             CV_MAT_DEPTH(CV_32FC3));
            d_tensor_output.step = up.width * up.height * CV_MAT_CN(CV_32FC3) * sizeof(BASE_CUDA_T(CV_32FC3));
            d_tensor_output.setTo(cv::Scalar(0.f, 0.f, 0.f));

            cv::Mat diff(up, CV_MAT_DEPTH(CV_32FC3));
            cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_32FC3), CV_MAT_DEPTH(CV_32FC3));
            

            std::array<cv::cuda::GpuMat, BATCH> crops;
            d_up.setTo(val_init_output);
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                crops[crop_i] = d_input(crops_2d[crop_i]);
                for (int i = 0; i < CV_MAT_CN(CV_8UC3); i++) {
                    h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_32FC3));
                }
            }

            // Setting up fk variables
            const std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> fk_crops{ cvGS::gpuMat2RawPtr2D_arr<uchar3, BATCH>(crops) };
            const fk::Size fk_size{up.width, up.height};
            cv::Mat h_fk_output(BATCH, up.width * up.height * CV_MAT_CN(CV_32FC3), CV_MAT_DEPTH(CV_32FC3));
            cv::cuda::GpuMat d_fk_output(BATCH, up.width * up.height * CV_MAT_CN(CV_32FC3),
                CV_MAT_DEPTH(CV_32FC3));
            d_fk_output.step = up.width * up.height * CV_MAT_CN(CV_32FC3) * sizeof(BASE_CUDA_T(CV_32FC3));
            d_fk_output.setTo(cv::Scalar(0.f, 0.f, 0.f));
            auto fk_output = cvGS::gpuMat2Tensor<float>(d_fk_output, up, 3);

            const float3 fk_val_alpha = cvGS::cvScalar2CUDAV<CV_32FC3>::get(val_alpha);
            const float3 fk_val_sub = cvGS::cvScalar2CUDAV<CV_32FC3>::get(val_sub);
            const float3 fk_val_div = cvGS::cvScalar2CUDAV<CV_32FC3>::get(val_div);
            const float3 fk_defaultBackground = cvGS::cvScalar2CUDAV<CV_32FC3>::get(val_init_output);

            START_OCV_BENCHMARK
            // fk version
            PUSH_RANGE("Launching fk")
            const auto resizeOp = fk::PerThreadRead<fk::_2D, uchar3>::build(BATCH, fk_defaultBackground, fk_crops)
                                  .then(fk::ResizeRead<fk::INTER_LINEAR, fk::PRESERVE_AR>::build(fk_size, fk_defaultBackground));
            fk::executeOperations(stream, resizeOp,
                                  fk::ColorConversion<fk::COLOR_RGB2BGR, float3, float3>::build(),
                                  fk::Mul<float3>::build(fk_val_alpha),
                                  fk::Sub<float3>::build(fk_val_sub),
                                  fk::Div<float3>::build(fk_val_div),
                                  fk::TensorSplit<float3>::build({ fk_output }));
            POP_RANGE

            STOP_OCV_START_CVGS_BENCHMARK
            // cvGPUSpeedup
            PUSH_RANGE("Launching cvGS")
            cvGS::executeOperations(cv_stream,
                cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH, cvGS::PRESERVE_AR>(crops, up, BATCH, val_init_output),
                cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(),
                cvGS::multiply<CV_32FC3>(val_alpha),
                cvGS::subtract<CV_32FC3>(val_sub),
                cvGS::divide<CV_32FC3>(val_div),
                cvGS::split<CV_32FC3>(d_tensor_output, up));
            POP_RANGE
            STOP_CVGS_BENCHMARK
            d_tensor_output.download(h_tensor_output, cv_stream);
            d_fk_output.download(h_fk_output, cv_stream);

            cv_stream.waitForCompletion();
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                cv::Mat row = h_tensor_output.row(crop_i);
                cv::Mat row_fk = h_fk_output.row(crop_i);
                for (int i = 0; i < CV_MAT_CN(CV_8UC3); i++) {
                    int planeStart = i * up.width * up.height;
                    int planeEnd = ((i + 1) * up.width * up.height) - 1;
                    cv::Mat plane = row.colRange(planeStart, planeEnd);
                    cv::Mat plane_fk = row_fk.colRange(planeStart, planeEnd);
                    h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
                    h_cvResults[crop_i].push_back(cv::Mat(up.height, up.width, plane_fk.type(), plane_fk.data));
                }
            }

            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                for (int i = 0; i < CV_MAT_CN(CV_32FC3); i++) {
                    cv::Mat cvRes = h_cvResults[crop_i].at(i);
                    cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
                    diff = cv::abs(cvRes - cvGSRes);
                    bool passedThisTime = checkResults<CV_MAT_DEPTH(CV_32FC3)>(diff.cols, diff.rows, diff);
                    passed &= passedThisTime;
                }
            }
        }
        catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
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
                ss << "test_batchresize_aspectratio_x_split3D<" << cvTypeToString<CV_8UC3>() << ", " << cvTypeToString<CV_32FC3>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_batchresize_aspectratio_x_split3D<" << cvTypeToString<CV_8UC3>() << ", " << cvTypeToString<CV_32FC3>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}
#endif

int launch() {
#ifdef ENABLE_BENCHMARK
    cv::cuda::Stream cv_stream;
    bool correct = test_batchresize_aspectratio_x_split3D<50>(cv_stream, true);

    return correct ? 0 : -1;
#else
    return 0;
#endif
}
