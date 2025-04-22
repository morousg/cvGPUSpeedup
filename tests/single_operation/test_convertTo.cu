/* Copyright 2025 Grup Mediapro S.L.U.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>
#include <tests/testsCommon.cuh>

#include <cvGPUSpeedup.cuh>

int launch() {
    cv::cuda::Stream stream;

    cv::cuda::GpuMat var1(cv::Size(16, 16), CV_8UC1);
    cv::cuda::GpuMat var2(cv::Size(16, 16), CV_8UC2);
    cv::cuda::GpuMat var3(cv::Size(16, 16), CV_8UC3);
    cv::cuda::GpuMat var4(cv::Size(16, 16), CV_8UC4);
    cv::cuda::GpuMat var4_2(cv::Size(16, 16), CV_8UC4);

    cv::cuda::GpuMat fvar1(cv::Size(16, 16), CV_32FC1);
    cv::cuda::GpuMat fvar2(cv::Size(16, 16), CV_32FC2);
    cv::cuda::GpuMat fvar3(cv::Size(16, 16), CV_32FC3);
    cv::cuda::GpuMat S32var4(cv::Size(16, 16), CV_32SC4);
    cv::cuda::GpuMat S32var4_2(cv::Size(16, 16), CV_32SC4);

    cv::cuda::GpuMat cvGSvar1(cv::Size(16, 16), CV_8UC1);
    cv::cuda::GpuMat cvGSvar2(cv::Size(16, 16), CV_8UC2);
    cv::cuda::GpuMat cvGSvar3(cv::Size(16, 16), CV_8UC3);
    cv::cuda::GpuMat cvGSvar4(cv::Size(16, 16), CV_8UC4);
    cv::cuda::GpuMat cvGSvar4_2(cv::Size(16, 16), CV_8UC4);

    cv::cuda::GpuMat cvGSfvar1(cv::Size(16, 16), CV_32FC1);
    cv::cuda::GpuMat cvGSfvar2(cv::Size(16, 16), CV_32FC2);
    cv::cuda::GpuMat cvGSfvar3(cv::Size(16, 16), CV_32FC3);
    cv::cuda::GpuMat cvGS32var4(cv::Size(16, 16), CV_32SC4);
    cv::cuda::GpuMat cvGS32var4_2(cv::Size(16, 16), CV_32SC4);

    var1.setTo(cv::Scalar(20));
    cvGSvar1.setTo(cv::Scalar(20));
    var2.setTo(cv::Scalar(20, 30));
    cvGSvar2.setTo(cv::Scalar(20, 30));
    var3.setTo(cv::Scalar(20, 30, 40));
    cvGSvar3.setTo(cv::Scalar(20, 30, 40));
    var4.setTo(cv::Scalar(20, 30, 40, 50));
    cvGSvar4.setTo(cv::Scalar(20, 30, 40, 50));
    var4_2.setTo(cv::Scalar(20, 30, 40, 50));
    cvGSvar4_2.setTo(cv::Scalar(20, 30, 40, 50));

    // const auto incorrectTypes = cvGS::convertTo<CV_8UC1, CV_32FC2>(); will fail compilation

    var1.convertTo(fvar1, CV_32FC1, stream);
    cvGS::executeOperations(cvGSvar1, cvGSfvar1, stream, cvGS::convertTo<CV_8UC1, CV_32FC1>());

    var2.convertTo(fvar2, CV_32FC2, stream);
    cvGS::executeOperations(cvGSvar2, cvGSfvar2, stream, cvGS::convertTo<CV_8UC2, CV_32FC2>());

    var3.convertTo(fvar3, CV_32FC3, 0.5, 0.5, stream);
    cvGS::executeOperations(cvGSvar3, cvGSfvar3, stream, cvGS::convertTo<CV_8UC3, CV_32FC3>(0.5f, 0.5f));

    var4.convertTo(S32var4, CV_32SC4, 0.5, 0.5, stream);
    cvGS::executeOperations(cvGSvar4, cvGS32var4, stream, cvGS::convertTo<CV_8UC4, CV_32SC4>(0.5f, 0.5f));

    var4_2.convertTo(S32var4_2, CV_32SC4, 0.5, stream);
    cvGS::executeOperations(cvGSvar4_2, cvGS32var4_2, stream, cvGS::convertTo<CV_8UC4, CV_32SC4>(0.5f));

    stream.waitForCompletion();

    // Download the results to host
    cv::Mat h_fvar1(fvar1);
    cv::Mat h_cvGSfvar1(cvGSfvar1);
    cv::Mat h_fvar2(fvar2);
    cv::Mat h_cvGSfvar2(cvGSfvar2);
    cv::Mat h_fvar3(fvar3);
    cv::Mat h_cvGSfvar3(cvGSfvar3);
    cv::Mat h_S32var4(S32var4);
    cv::Mat h_cvGS32var4(cvGS32var4);
    cv::Mat h_S32var4_2(S32var4_2);
    cv::Mat h_cvGS32var4_2(cvGS32var4_2);

    // Compare
    double normValue = cv::norm(h_fvar1, h_cvGSfvar1);
    double normValue2 = cv::norm(h_fvar2, h_cvGSfvar2);
    double normValue3 = cv::norm(h_fvar3, h_cvGSfvar3);
    double normValue4 = cv::norm(h_S32var4, h_cvGS32var4);
    double normValue5 = cv::norm(h_S32var4_2, h_cvGS32var4_2);
    // Check if the matrices are identical
    if (normValue == 0 && normValue2 == 0 && normValue3 == 0 && normValue4 == 0 && normValue5 == 0) {
        std::cout << "The matrices are identical." << std::endl;
    } else {
        std::cout << "The matrices are different. Norm value: " << normValue << std::endl;
    }

    return 0;
}