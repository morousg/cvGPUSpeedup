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

#include "testUtils.cuh"
#include <cv2cuda_types.cuh>

#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

template <int T>
bool checkResults(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& h_comparison1C) {
    cv::Mat h_comparison(NUM_ELEMS_Y, NUM_ELEMS_X, T);
    cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, T, 0.00001);
    cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_GT);

#ifdef CVGS_DEBUG
    for (int y = 0; y < h_comparison1C.rows; y++) {
        for (int x = 0; x < h_comparison1C.cols; x++) {
            CUDA_T(T) value = h_comparison1C.at<CUDA_T(T)>(y, x);
            if (value > 0.00001) {
                std::cout << "(" << x << "," << y << ")= " << value << ";" << std::endl;
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

    for (int i = 0; i < CV_MAT_CN(T); i++) {
        passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C.at(i));
    }
    return passed;
}

// Uncomment to enable benchmark
//#define ENABLE_BENCHMARK

#ifdef ENABLE_BENCHMARK
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map < std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 1000;

struct BenchmarkResultsNumbers {
    float OCVelapsedTimeMax;
    float OCVelapsedTimeAcum;
    float cvGSelapsedTimeMax;
    float cvGSelapsedTimeAcum;
};

template <size_t ITERATIONS>
float computeVariance(const float& mean, const std::array<float, ITERATIONS>& times) {
    float sumOfDiff = 0.f;
    for (int i = 0; i < ITERATIONS; i++) {
        const float diff = times[i] - mean;
        sumOfDiff += (diff * diff);
    }
    return sumOfDiff / (ITERATIONS - 1);
}

template <int CV_INPUT_TYPE, int CV_OUTPUT_TYPE, int BATCH, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<int, NUM_BATCH_VALUES>& batchValues>
void processExecution(const BenchmarkResultsNumbers& resF, const std::string& functionName,
    const std::array<float, ITERS>& OCVelapsedTime, const std::array<float, ITERS>& cvGSelapsedTime) {
    if constexpr (BATCH == 1) {
        const std::string fileName = functionName + std::string(".csv");
        if (currentFile.find(fileName) == currentFile.end()) {
            currentFile[fileName].open(path + fileName);
            currentFile[fileName] << "Number of images";
            for (const auto& i : batchValues) {
                currentFile[fileName] << ", Avg" << i << ", Var" << i;
            }
            currentFile[fileName] << "\n";
        }
        benchmarkResultsText.clear();
        benchmarkResultsText["SpeedupLine"] << "Speedup " << cvTypeToString<CV_INPUT_TYPE>() << "_" << cvTypeToString<CV_OUTPUT_TYPE>();
    }
    const float ocvMean = resF.OCVelapsedTimeAcum / ITERATIONS;
    const float cvgsMean = resF.cvGSelapsedTimeAcum / ITERATIONS;
    const float ocvVariance = computeVariance(ocvMean, OCVelapsedTime);
    const float cvgsVariance = computeVariance(cvgsMean, cvGSelapsedTime);

    benchmarkResultsText["SpeedupLine"] << ", " << (ocvMean / cvgsMean);
    benchmarkResultsText["SpeedupLine"] << ", " << (ocvVariance / cvgsVariance);

    if constexpr (BATCH == batchValues[NUM_BATCH_VALUES - 1]) {
        const std::string fileName = functionName + std::string(".csv");
        currentFile[fileName] << benchmarkResultsText["SpeedupLine"].str() << std::endl;
    }
}

#endif

#ifdef ENABLE_BENCHMARK
#define START_OCV_BENCHMARK \
cudaEvent_t start, stop; \
BenchmarkResultsNumbers resF{0.f, 0.f, 0.f, 0.f}; \
cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream); \
gpuErrchk(cudaEventCreate(&start)); \
gpuErrchk(cudaEventCreate(&stop)); \
std::array<float, ITERS> OCVelapsedTime; \
std::array<float, ITERS> cvGSelapsedTime; \
for (int i = 0; i < ITERS; i++) { \
    gpuErrchk(cudaEventRecord(start, stream));
#else
#define START_OCV_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_OCV_START_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&OCVelapsedTime[i], start, stop)); \
resF.OCVelapsedTimeMax = resF.OCVelapsedTimeMax < OCVelapsedTime[i] ? OCVelapsedTime[i] : resF.OCVelapsedTimeMax; \
resF.OCVelapsedTimeAcum += OCVelapsedTime[i]; \
gpuErrchk(cudaEventRecord(start, stream));
#else
#define STOP_OCV_START_CVGS_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&cvGSelapsedTime[i], start, stop)); \
resF.cvGSelapsedTimeMax = resF.cvGSelapsedTimeMax < cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMax; \
resF.cvGSelapsedTimeAcum += cvGSelapsedTime[i]; \
} \
processExecution<CV_TYPE_I, CV_TYPE_O, BATCH, ITERS, batchValues.size(), batchValues>(resF, __func__, OCVelapsedTime, cvGSelapsedTime);
#else
#define STOP_CVGS_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define CLOSE_BENCHMARK \
for (auto&& [_, file] : currentFile) { \
    file.close(); \
}
#else
#define CLOSE_BENCHMARK
#endif