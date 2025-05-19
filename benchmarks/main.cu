#include "mul_launcher.h" // Will be found via include_directories
#include <iostream>
// #include <opencv2/opencv.hpp> // For actual cv::cuda::GpuMat etc.
// --- Dummy/Placeholder types and constants for standalone compilation ---
// --- End Dummies ---
int main() {
  // Dummy data for the function call
  std::array<cv::cuda::GpuMat, REAL_BATCH> crops_data;
  cv::cuda::Stream stream_data;
  float alpha_data = 1.0f;
  cv::cuda::GpuMat output_tensor_data;
  cv::Size crop_size_data = {0,0};
  MulFuncType func_data;
  std::cout << "Launching experiment for N=1 via Mul Pipeline..." << std::endl;
  launchMulPipeline(crops_data, stream_data, alpha_data, output_tensor_data, crop_size_data, func_data);
  if (NUM_EXPERIMENTS_AS_INT >= 2) { // Use the value from CMake if needed for runtime loops
    std::cout << "Launching experiment for N=2 via Mul Pipeline..." << std::endl;
    launchMulPipeline(crops_data, stream_data, alpha_data, output_tensor_data, crop_size_data, func_data);
  }
  // ... and so on, or use a loop if you pass N at runtime (which launchMulPipeline is not designed for)
  cudaDeviceSynchronize(); // Wait for all CUDA work to finish
  std::cout << "All launched experiments synchronized." << std::endl;
  return 0;
}