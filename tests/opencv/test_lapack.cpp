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

#ifndef FK_TEST_LAPACK
#define FK_TEST_LAPACK

#include "tests/main.h"

#include <fused_kernel/algorithms/image_processing/warping.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

int launch() {
    std::vector<cv::Point2f> srcPoints = { {0, 0}, {300, 0}, {300, 300}, {0, 300} };
    std::vector<cv::Point2f> dstPoints = { {50, 50}, {250, 50}, {250, 250}, {50, 250} };

    cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    constexpr std::array<fk::Point_<float, fk::_2D>, 4> srcPointsArray = { fk::Point_<float, fk::_2D>{0, 0}, {300, 0}, {300, 300}, {0, 300} };
    constexpr std::array<fk::Point_<float, fk::_2D>, 4> dstPointsArray = { fk::Point_<float, fk::_2D>{50, 50}, {250, 50}, {250, 250}, {50, 250} };

    fk::Warping<float, fk::WarpType::Perspective,
        fk::Read<fk::PerThreadRead<fk::_2D, float>>>::build(
        srcPointsArray, dstPointsArray, fk::DecompTypes::DECOMP_LU);

    return 0;
}

#endif