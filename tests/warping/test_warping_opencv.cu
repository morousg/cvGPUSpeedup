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
#include <opencv2/opencv.hpp>

bool testPerspective() {
    // Load the image
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Upload the image to GPU
    cv::cuda::GpuMat d_img;
    d_img.upload(img);

    // Define the source and destination points for perspective transformation
    cv::Point2f src_points[4] = { cv::Point2f(56, 65), cv::Point2f(368, 52), cv::Point2f(28, 387), cv::Point2f(389, 390) };
    cv::Point2f dst_points[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    // Get the perspective transformation matrix
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // Upload the transformation matrix to GPU
    cv::cuda::GpuMat d_perspective_matrix;
    d_perspective_matrix.upload(perspective_matrix);

    // Apply the perspective transformation
    cv::cuda::GpuMat d_result;
    cv::cuda::warpPerspective(d_img, d_result, perspective_matrix, img.size());

    // Download the result back to CPU
    cv::Mat result(d_result);

    return true;
}

bool testAffine() {
    // Load the image
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Upload the image to GPU
    cv::cuda::GpuMat d_img;
    d_img.upload(img);

    // Define the translation values
    double tx = 50, ty = 100;

    // Get the affine transformation matrix
    cv::Mat affine_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);

    // Upload the transformation matrix to GPU
    cv::cuda::GpuMat d_affine_matrix;
    d_affine_matrix.upload(affine_matrix);

    // Apply the affine transformation
    cv::cuda::GpuMat d_result;
    cv::cuda::warpAffine(d_img, d_result, affine_matrix, img.size());

    // Download the result back to CPU
    cv::Mat result(d_result);

    return true;
}

int launch() {
    return testPerspective() && testAffine() ? 0 : -1;
}
