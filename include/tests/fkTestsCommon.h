/* Copyright 2024 Oscar Amoros Huguet

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
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <array>

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
    return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence = generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});

#ifdef ENABLE_BENCHMARK
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;

struct BenchmarkResultsNumbers {
    float cvGSelapsedTimeMax;
    float cvGSelapsedTimeMin;
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

template <int BATCH, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES>& batchValues>
inline void processExecution(const BenchmarkResultsNumbers& resF,
                             const std::string& functionName,
                             const std::array<float, ITERS>& cvGSelapsedTime,
                             const std::string& variableDimension) {

    // Create 2D Table for changing types and changing batch
    const std::string fileName = functionName + std::string(".csv");
    if constexpr (BATCH == batchValues[0]) {
        if (currentFile.find(fileName) == currentFile.end()) {
            currentFile[fileName].open(path + fileName);
        }
        currentFile[fileName] << variableDimension;
        currentFile[fileName] << ", MeanTime";
        currentFile[fileName] << ", TimeVariance";
        currentFile[fileName] << ", MaxTime";
        currentFile[fileName] << ", MinTime";
        currentFile[fileName] << std::endl;
    }

    const bool mustStore = currentFile.find(fileName) != currentFile.end();
    if (mustStore) {
        const float cvgsMean = resF.cvGSelapsedTimeAcum / ITERATIONS;
        const float cvgsVariance = computeVariance(cvgsMean, cvGSelapsedTime);

        currentFile[fileName] << BATCH;
        currentFile[fileName] << ", " << cvgsMean;
        currentFile[fileName] << ", " << computeVariance(cvgsMean, cvGSelapsedTime);
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMax;
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMin;
        currentFile[fileName] << std::endl;
    }
}
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_OCV_START_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(start, stream));
#else
#define START_CVGS_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&cvGSelapsedTime[i], start, stop)); \
resF.cvGSelapsedTimeMax = resF.cvGSelapsedTimeMax < cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMax; \
resF.cvGSelapsedTimeMin = resF.cvGSelapsedTimeMin > cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMin; \
resF.cvGSelapsedTimeAcum += cvGSelapsedTime[i]; \
} \
processExecution<BATCH, ITERS, batchValues.size(), batchValues>(resF, __func__, cvGSelapsedTime, VARIABLE_DIMENSION);
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