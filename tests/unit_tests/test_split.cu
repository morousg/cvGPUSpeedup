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

template <size_t N>
cv::cuda::GpuMat initInput() {
    cv::Mat h_input(cv::Size(16, 16), CV_8UC3, cv::Scalar(1, 2, 3));
    cv::cuda::GpuMat d_input;
    d_input.upload(h_input);
    return d_input;
}

template <size_t... Idx>
std::array<cv::cuda::GpuMat, sizeof...(Idx)> initInputArray(const std::index_sequence<Idx...>&) {
    return { initInput<Idx>()... };
}

template <size_t N>
std::vector<cv::cuda::GpuMat> initOutput() {
    std::vector<cv::cuda::GpuMat> output;
    for (int c = 0; c < 3; ++c) {
        output.emplace_back(cv::Size(16, 16), CV_8UC1);
    }
    return output;
}

template <size_t... Idx>
std::array<std::vector<cv::cuda::GpuMat>, sizeof...(Idx)> initOutputArray(const std::index_sequence<Idx...>&) {
    return { initOutput<Idx>()... };
}

bool testSplitBatch() {
    cv::cuda::Stream stream;
    constexpr size_t BATCH = 10;
    
    std::array<cv::cuda::GpuMat, BATCH> inputArr = initInputArray(std::make_index_sequence<BATCH>{});
    std::array<std::vector<cv::cuda::GpuMat>, BATCH> outputArr = initOutputArray(std::make_index_sequence<BATCH>{});

    const auto splitIOp = cvGS::split<CV_8UC3>(outputArr);

    cvGS::executeOperations(inputArr, stream, splitIOp);
    for (int i = 0; i < BATCH; ++i) {
        for (int c = 0; c < 3; ++c) {
            cv::Mat h_output;
            outputArr[i][c].download(h_output, stream);
            stream.waitForCompletion();
            if (!cv::countNonZero(h_output == cv::Scalar(1 + c))) {
                std::cerr << "Error in split operation at batch " << i << ", channel " << c << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool testSplit() {
    cv::cuda::Stream stream;
    
    cv::cuda::GpuMat input = initInput<0>();
    std::vector<cv::cuda::GpuMat> output = initOutput<0>();

    const auto splitIOp = cvGS::split<CV_8UC3>(output);

    cvGS::executeOperations(input, stream, splitIOp);
    for (int c = 0; c < 3; ++c) {
        cv::Mat h_output;
        output[c].download(h_output, stream);
        stream.waitForCompletion();
        if (!cv::countNonZero(h_output == cv::Scalar(1 + c))) {
            std::cerr << "Error in split operation at channel " << c << std::endl;
            return false;
        }
    }
    return true;
}

int launch() {
    bool splitBatch = testSplitBatch();
    bool split = testSplit();

    if (!splitBatch || !split) {
        if (!splitBatch) {
            std::cerr << "Split batch test failed!" << std::endl;
        }
        if (!split) {
            std::cerr << "Split test failed!" << std::endl;
        }
        return -1;
    }
    std::cout << "Split tests passed successfully!" << std::endl;
    return 0;
}