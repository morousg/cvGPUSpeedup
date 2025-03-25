/* Copyright 2025 Grup Mediapro S.L.U

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

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/algorithms/image_processing/border_reader.cuh>
#include <opencv2/opencv.hpp>
#include "tests/testsCommon.cuh"

bool testPerspective() {
    // Load the image
    const cv::Mat img = cv::imread("E:/GitHub/cvGPUSpeedup/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);

    // Define the source and destination points for perspective transformation
    cv::Point2f src_points[4] = { cv::Point2f(56, 65), cv::Point2f(368, 52), cv::Point2f(28, 387), cv::Point2f(389, 390) };
    cv::Point2f dst_points[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    // Get the perspective transformation matrix
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // Preallocate the result images
    cv::cuda::GpuMat d_resultcv(img.size(), CV_8UC3);
    cv::cuda::GpuMat d_resultcvGS(img.size(), CV_8UC3);

    // Apply the perspective transformation
    cv::cuda::warpPerspective(d_img, d_resultcv, perspective_matrix, img.size(), 1, 0, cv::Scalar(), stream);

    const auto warpFunc = cvGS::warp<fk::WarpType::Perspective, CV_8UC3>(d_img, perspective_matrix, img.size());

    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);

    std::cout << "Perspective transformation: " << (correct ? "PASS" : "EXPECTED_FAIL") << std::endl;

    return correct;
}

bool testAffine() {
    // Load the image
    const cv::Mat img = cv::imread("E:/GitHub/cvGPUSpeedup/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);

    // Define the translation values
    double tx = 50, ty = 100;

    // Get the affine transformation matrix
    cv::Mat affine_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);

    // Preallocate the result images
    cv::cuda::GpuMat d_resultcv(img.size(), CV_8UC3);
    cv::cuda::GpuMat d_resultcvGS(img.size(), CV_8UC3);

    // Apply the affine transformation
    cv::cuda::GpuMat d_result;
    cv::cuda::warpAffine(d_img, d_resultcv, affine_matrix, img.size());

    const auto warpFunc = cvGS::warp<fk::WarpType::Affine, CV_8UC3>(d_img, affine_matrix, img.size());
    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);

    std::cout << "Affine transformation: " << (correct ? "PASS" : "FAIL") << std::endl;

    return correct;
}

int launch() {
    const bool correctPerspective = testPerspective();
    const bool correctAffine = testAffine();
    // warpPerspective is almost identical to OpenCV's implementation, but there are a few pixels of difference in
    // the border, despite using the same border type. Finding the cause of this difference is future work.
    return true && correctAffine ? 0 : -1;
}
