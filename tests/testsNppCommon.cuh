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

#include "tests/testUtils.cuh"
#include <fused_kernel/core/utils/vlimits.h>

#include <array>
#include <fstream>
#include <sstream>
#include <unordered_map>

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
  return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence =
    generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});
 
 

#ifdef ENABLE_BENCHMARK
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{""};

constexpr int ITERS = 100;

struct BenchmarkResultsNumbers {
  float NPPelapsedTimeMax;
  float NPPelapsedTimeMin;
  float NPPelapsedTimeAcum;
  float FKelapsedTimeMax;
  float FKelapsedTimeMin;
  float FKelapsedTimeAcum;
};

template <size_t ITERATIONS> float computeVariance(const float &mean, const std::array<float, ITERATIONS> &times) {
  float sumOfDiff = 0.f;
  for (int i = 0; i < ITERATIONS; i++) {
    const float diff = times[i] - mean;
    sumOfDiff += (diff * diff);
  }
  return sumOfDiff / (ITERATIONS - 1);
}

template <int VARIABLE_DIMENSION, int ITERATIONS, int NUM_BATCH_VALUES,
          const std::array<size_t, NUM_BATCH_VALUES> &variableDimanesionValues>
inline void processExecution(const BenchmarkResultsNumbers &resF, const std::string &functionName,
                             const std::array<float, ITERS> &NPPelapsedTime,
                             const std::array<float, ITERS> &FKelapsedTime, const std::string &variableDimension) {
  // Create 2D Table for changing types and changing batch
  const std::string fileName = functionName + std::string(".csv");
  if constexpr (VARIABLE_DIMENSION == variableDimanesionValues[0]) {
    if (currentFile.find(fileName) == currentFile.end()) {
      currentFile[fileName].open(path + fileName);
    }
    currentFile[fileName] << variableDimension;
    currentFile[fileName] << ", NPP MeanTime";
    currentFile[fileName] << ", NPP TimeVariance";
    currentFile[fileName] << ", NPP MaxTime";
    currentFile[fileName] << ", NPP MinTime";
    currentFile[fileName] << ", FK MeanTime";
    currentFile[fileName] << ", FK TimeVariance";
    currentFile[fileName] << ", FK MaxTime";
    currentFile[fileName] << ", FK MinTime";
    currentFile[fileName] << ", Mean Speedup";
    currentFile[fileName] << std::endl;
  }

  const bool mustStore = currentFile.find(fileName) != currentFile.end();
  if (mustStore) {
    const float NPPMean = resF.NPPelapsedTimeAcum / ITERATIONS;
    const float FKMean = resF.FKelapsedTimeAcum / ITERATIONS;
    const float NPPVariance = computeVariance(NPPMean, NPPelapsedTime);
    const float FKVariance = computeVariance(FKMean, FKelapsedTime);
    float meanSpeedup{0.f};
    for (int i = 0; i < ITERS; i++) {
      meanSpeedup += NPPelapsedTime[i] / FKelapsedTime[i];
    }
    meanSpeedup /= ITERS;

    currentFile[fileName] << VARIABLE_DIMENSION;
    currentFile[fileName] << BATCH;
    currentFile[fileName] << ", " << NPPMean;
    currentFile[fileName] << ", " << computeVariance(NPPMean, NPPelapsedTime);
    currentFile[fileName] << ", " << resF.NPPelapsedTimeMax;
    currentFile[fileName] << ", " << resF.NPPelapsedTimeMin;
    currentFile[fileName] << ", " << FKMean;
    currentFile[fileName] << ", " << computeVariance(FKMean, FKelapsedTime);
    currentFile[fileName] << ", " << resF.FKelapsedTimeMax;
    currentFile[fileName] << ", " << resF.FKelapsedTimeMin;
    currentFile[fileName] << ", " << meanSpeedup;
    currentFile[fileName] << std::endl;
  }
}
#endif 
 
#ifdef ENABLE_BENCHMARK
#define START_NPP_BENCHMARK                                                                                            \
  std::cout << "Executing " << __func__ << " fusing " << BATCH << " operations. " << (BATCH - FIRST_VALUE) / INCREMENT \
            << "/" << NUM_EXPERIMENTS << std::endl;                                                                    \
  cudaEvent_t start, stop;                                                                                             \
  BenchmarkResultsNumbers resF;                                                                                        \
  resF.NPPelapsedTimeMax = fk::minValue<float>;                                                                        \
  resF.NPPelapsedTimeMin = fk::maxValue<float>;                                                                        \
  resF.NPPelapsedTimeAcum = 0.f;                                                                                       \
  resF.FKelapsedTimeMax = fk::minValue<float>;                                                                         \
  resF.FKelapsedTimeMin = fk::maxValue<float>;                                                                         \
  resF.FKelapsedTimeAcum = 0.f;                                                                                        \
                                               \
  gpuErrchk(cudaEventCreate(&start));                                                                                  \
  gpuErrchk(cudaEventCreate(&stop));                                                                                   \
  std::array<float, ITERS> NPPelapsedTime;                                                                             \
  std::array<float, ITERS> FKelapsedTime;                                                                              \
  for (int i = 0; i < ITERS; i++) {                                                                                    \
    gpuErrchk(cudaEventRecord(start, compute_stream));
#else
#define START_NPP_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_NPP_START_FK_BENCHMARK                                                                                    \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&NPPelapsedTime[i], start, stop));                                                    \
  resF.NPPelapsedTimeMax = resF.NPPelapsedTimeMax < NPPelapsedTime[i] ? NPPelapsedTime[i] : resF.NPPelapsedTimeMax;    \
  resF.NPPelapsedTimeMin = resF.NPPelapsedTimeMin > NPPelapsedTime[i] ? NPPelapsedTime[i] : resF.NPPelapsedTimeMin;    \
  resF.NPPelapsedTimeAcum += NPPelapsedTime[i];                                                                        \
  gpuErrchk(cudaEventRecord(start, compute_stream));
#else
#define STOP_NPP_START_FK_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_FK_BENCHMARK                                                                                              \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&FKelapsedTime[i], start, stop));                                                     \
  resF.FKelapsedTimeMax = resF.FKelapsedTimeMax < FKelapsedTime[i] ? FKelapsedTime[i] : resF.FKelapsedTimeMax;         \
  resF.FKelapsedTimeMin = resF.FKelapsedTimeMin > FKelapsedTime[i] ? FKelapsedTime[i] : resF.FKelapsedTimeMin;         \
  resF.FKelapsedTimeAcum += FKelapsedTime[i];                                                                          \
  } \                                                                                                                \
processExecution<BATCH, ITERS, batchValues.size(), batchValues>(                               \
      resF, __func__, NPPelapsedTime, FKelapsedTime, VARIABLE_DIMENSION);
#else
#define STOP_FK_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define CLOSE_BENCHMARK                                                                                                \
  for (auto &&[_, file] : currentFile) {                                                                               \
    file.close();                                                                                                      \
  }
#else
#define CLOSE_BENCHMARK
#endif