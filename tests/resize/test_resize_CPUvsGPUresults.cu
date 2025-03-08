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

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <cvGPUSpeedup.cuh>

int launch() {
    cv::cuda::Stream cv_stream;

    // Load the PNG file
    const std::string filePath{ "" };
    cv::Mat image = cv::imread(filePath.c_str(), cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (image.empty())
    {
        std::cerr << "test_resize_CPUvsGPUresults SKIPPED. No valid image file path probided." << std::endl;
        return 0;
    }

    const cv::Size resizeSize(1920, 1080);
    const auto interpolationType = cv::INTER_LINEAR;
    cv::Mat biggerCPU;
    cv::resize(image, biggerCPU, resizeSize, 0., 0., interpolationType);

    cv::cuda::GpuMat imageGPU(image);
    cv::cuda::GpuMat outputGPU;
    cv::cuda::GpuMat outputcvGS(resizeSize, CV_8UC4);
    cv::cuda::resize(imageGPU, outputGPU, resizeSize, 0., 0., interpolationType, cv_stream);
    cvGS::executeOperations(cv_stream, cvGS::resize<CV_8UC4, cv::INTER_LINEAR>(imageGPU, resizeSize, 0., 0.),
                                       cvGS::convertTo<CV_32FC4, CV_8UC4>(),
                                       cvGS::write<CV_8UC4>(outputcvGS));

    cv::Mat outputGPU_down;
    outputGPU.download(outputGPU_down, cv_stream);
    cv::Mat outputcvGS_down;
    outputcvGS.download(outputcvGS_down, cv_stream);

    cv_stream.waitForCompletion();

    cv::Mat CPUUINT;
    cv::Mat GPUUINTOCV;
    cv::Mat CVGSUINT;
    biggerCPU.convertTo(CPUUINT, CV_32SC4);
    outputGPU_down.convertTo(GPUUINTOCV, CV_32SC4);
    outputcvGS_down.convertTo(CVGSUINT, CV_32SC4);

    cv::Mat diffOCV;
    cv::absdiff(CPUUINT, GPUUINTOCV, diffOCV);
    diffOCV.convertTo(diffOCV, CV_8UC4);
    diffOCV *= 5;
    diffOCV += cv::Scalar(0, 0, 0, 255);

    cv::Mat diffcvGS;
    cv::absdiff(CPUUINT, CVGSUINT, diffcvGS);
    diffcvGS.convertTo(diffcvGS, CV_8UC4);
    diffcvGS *= 5;
    diffcvGS += cv::Scalar(0, 0, 0, 255);

    cv::Mat diffOCVG_cvGS;
    cv::absdiff(GPUUINTOCV, CVGSUINT, diffOCVG_cvGS);
    diffOCVG_cvGS.convertTo(diffOCVG_cvGS, CV_8UC4);
    diffOCVG_cvGS += cv::Scalar(0, 0, 0, 255);
    cv::Scalar total = cv::sum(diffOCVG_cvGS);

    cv::Scalar minValOCV, maxValOCV;
    std::vector<cv::Mat> channelsOCV(4);
    cv::split(diffOCV, channelsOCV);
    for (int i = 0; i < 4; i++) {
        cv::minMaxLoc(channelsOCV[i], &minValOCV[i], &maxValOCV[i]);
    }

    cv::Scalar minValcvGS, maxValcvGS;
    std::vector<cv::Mat> channelscvGS(4);
    cv::split(diffcvGS, channelscvGS);
    for (int i = 0; i < 4; i++) {
        cv::minMaxLoc(channelscvGS[i], &minValcvGS[i], &maxValcvGS[i]);
    }

    return 0;
}
