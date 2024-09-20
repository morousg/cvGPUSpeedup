/* Copyright 2024 Oscar Amoros Huguet
   Copyright 2024 Albert Andaluz Gonzalez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <npp.h>
#include <nppi_geometry_transforms.h>

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/image_processing/resize_builders.cuh>
#include <fused_kernel/fused_kernel.cuh>
#include <numeric>
#include <sstream>

#include "tests/main.h"
#include "tests/testUtils.cuh"
#include "tests/testsNppCommon.cuh"
constexpr char VARIABLE_DIMENSION[]{"Batch size"};
constexpr size_t NUM_EXPERIMENTS = 9;
constexpr size_t FIRST_VALUE = 10;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;
enum CHANNEL { RED, GREEN, BLUE };
constexpr inline void nppAssert(NppStatus code, const char *file, int line, bool abort = true) {
  if (code != NPP_SUCCESS) {
    std::cout << "NPP failure: "
              << " File: " << file << " Line:" << line << std::endl;
    if (abort)
      throw std::exception();
  }
}

#define NPP_CHECK(ans)                                                                                                 \
  { nppAssert((ans), __FILE__, __LINE__, true); }

NppStreamContext initNppStreamContext(const cudaStream_t &stream);
NppStreamContext initNppStreamContext(const cudaStream_t &stream) {
  int device = 0;
  int ccmajor = 0;
  int ccminor = 0;
  uint flags;

  cudaDeviceProp prop;
  gpuErrchk(cudaGetDevice(&device)) gpuErrchk(cudaGetDeviceProperties(&prop, device));
  gpuErrchk(cudaDeviceGetAttribute(&ccmajor, cudaDevAttrComputeCapabilityMinor, device));
  gpuErrchk(cudaDeviceGetAttribute(&ccminor, cudaDevAttrComputeCapabilityMajor, device));
  gpuErrchk(cudaStreamGetFlags(stream, &flags));
  NppStreamContext nppstream = {stream,
                                device,
                                prop.multiProcessorCount,
                                prop.maxThreadsPerMultiProcessor,
                                prop.maxThreadsPerBlock,
                                prop.sharedMemPerBlock,
                                ccmajor,
                                ccminor,
                                flags};
  return nppstream;
}

template <int BATCH>
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cudaStream_t &compute_stream,
                                    bool enabled) {
  std::stringstream error_s;
  bool passed = true;
  bool exception = false;

  if (enabled) {
    const float alpha = 0.3f;
    const uchar CROP_W = 60;
    const uchar CROP_H = 120;
    const uchar UP_W = 64;
    const uchar UP_H = 128;
    try {
      constexpr uchar3 init_val{1, 2, 3};
      fk::Ptr2D<uchar3> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
      fk::setTo(init_val, d_input, compute_stream);
      fk::Ptr2D<float3> d_input_f(NUM_ELEMS_X, NUM_ELEMS_Y);
      std::array<fk::Ptr2D<float3>, BATCH> d_resized_npp, d_swap, d_mul, d_sub, d_div;
      std::array<fk::Ptr2D<float>, BATCH> d_channelA, d_channelB, d_channelC;
      std::array<fk::Ptr2D<float>, BATCH> h_channelA, h_channelB, h_channelC;
      fk::Tensor<float> d_tensornpp(UP_W, UP_H, BATCH, 3);
      NppiImageDescriptor *hBatchSrc = nullptr, *dBatchSrc = nullptr, *hBatchDst = nullptr, *dBatchDst = nullptr;
      NppiResizeBatchROI_Advanced *dBatchROI = nullptr, *hBatchROI = nullptr;

      // init data adn set to inital value
      NppStreamContext nppcontext = initNppStreamContext(compute_stream);

      // source images (rgb 8bit)

      const Npp32f mulValue[3] = {alpha, alpha, alpha};
      const Npp32f subValue[3] = {1.f, 4.f, 3.2f};
      const Npp32f divValue[3] = {1.f, 4.f, 3.2f};
      const NppiSize sz{UP_W, UP_H};

      fk::Tensor<float> h_tensor(UP_W, UP_H, BATCH, 3, fk::MemType::HostPinned);
      fk::Tensor<float> d_tensor(UP_W, UP_H, BATCH, 3);
      // crop array of images using batch resize+ ROIs
      // force fill 5
      constexpr int COLOR_PLANE = UP_H * UP_W;
      constexpr int IMAGE_STRIDE = (COLOR_PLANE * 3);
      // asume RGB->BGR
      const int aDstOrder[3] = {BLUE, GREEN, RED};
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

      for (int i = 0; i < BATCH; ++i) {
        hBatchSrc[i].pData = d_input_f.ptr().data;
        hBatchSrc[i].nStep = d_input_f.ptr().dims.pitch;
        hBatchSrc[i].oSize = NppiSize{static_cast<int>(NUM_ELEMS_X), static_cast<int>(NUM_ELEMS_Y)};
      }

      std::vector<std::array<Npp32f *, 3>> aDst;
      // dest images (Rgb, 32f)
      for (int i = 0; i < BATCH; ++i) {
        // NPP variables
        d_resized_npp[i] = fk::Ptr2D<float3>(UP_W, UP_H);
        d_swap[i] = fk::Ptr2D<float3>(UP_W, UP_H);
        d_mul[i] = fk::Ptr2D<float3>(UP_W, UP_H);
        d_sub[i] = fk::Ptr2D<float3>(UP_W, UP_H);
        d_div[i] = fk::Ptr2D<float3>(UP_W, UP_H);
        // Fill NPP Batch struct
        hBatchDst[i].pData = d_resized_npp[i].ptr().data;
        hBatchDst[i].nStep = d_resized_npp[i].ptr().dims.pitch;
        hBatchDst[i].oSize = sz;
        // Allocate pointers for split images (device)
        d_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H);
        // Allocate pointers for split images (host)
        h_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);

        std::array<Npp32f *, 3> ptrs = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};
        aDst.push_back(ptrs);
      }

      // ROI
      for (int i = 0; i < BATCH; ++i) {
        // NppiRect srcrect = {i, i, CROP_W, CROP_H};
        NppiRect srcrect = {i, i, CROP_W, CROP_H};
        NppiRect dstrect = {0, 0, UP_W, UP_H};
        hBatchROI[i].oSrcRectROI = srcrect;
        hBatchROI[i].oDstRectROI = dstrect;
      }

      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchROI), hBatchROI,
                                sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, compute_stream));

      std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> d_crop_fk;
      for (int i = 0; i < BATCH; i++) {
        d_crop_fk[i] = d_input.crop2D(fk::Point(i, i), fk::PtrDims<fk::_2D>(CROP_W, CROP_H));
      }

      // This operation parameters won't change, so we can generate them once instead of
      // generating them on evey iteration
      auto colorConvert = fk::Unary<fk::VectorReorder<float3, BLUE, GREEN, RED>>{};
      auto multiply = fk::Binary<fk::Mul<float3>>{fk::make_<float3>(mulValue[0], mulValue[1], mulValue[2])};
      auto sub = fk::Binary<fk::Sub<float3>>{fk::make_<float3>(subValue[0], subValue[1], subValue[2])};
      auto div = fk::Binary<fk::Div<float3>>{fk::make_<float3>(divValue[0], divValue[1], divValue[2])};

      START_NPP_BENCHMARK

      // NPP version
      // convert to 32f
      // print2D("Values after initialization", d_input, compute_stream);

      NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<Npp8u *>(d_input.ptr().data), d_input.ptr().dims.pitch,
                                          reinterpret_cast<Npp32f *>(d_input_f.ptr().data), d_input_f.ptr().dims.pitch,
                                          NppiSize{static_cast<int>(NUM_ELEMS_X), static_cast<int>(NUM_ELEMS_Y)},
                                          nppcontext));

      // print2D("Values after conversion to fp32", d_input_f, compute_stream);

      NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dBatchSrc, dBatchDst, dBatchROI, BATCH,
                                                     NPPI_INTER_LINEAR, nppcontext));

      /*for (int i = 0; i < BATCH; ++i) {
        std::stringstream ss;
        ss << "Resized plane " << i;
        print2D(ss.str(), d_resized_npp[i], compute_stream);
      }*/

      for (int i = 0; i < BATCH; ++i) {
        // std::cout << "Processing BATCH " << i << std::endl;
        NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(
            reinterpret_cast<Npp32f *>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch,
            reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].dims().pitch, sz, aDstOrder, nppcontext));

        // print2D("Values after swap", d_swap[i], compute_stream);

        NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].ptr().dims.pitch,
                                       mulValue, reinterpret_cast<Npp32f *>(d_mul[i].ptr().data),
                                       d_mul[i].ptr().dims.pitch, sz, nppcontext));

        // print2D("Values after mul", d_mul[i], compute_stream);

        NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch,
                                       subValue, reinterpret_cast<Npp32f *>(d_sub[i].ptr().data),
                                       d_sub[i].ptr().dims.pitch, sz, nppcontext));

        // print2D("Values after sub", d_sub[i], compute_stream);

        NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch,
                                       divValue, reinterpret_cast<Npp32f *>(d_div[i].ptr().data),
                                       d_div[i].ptr().dims.pitch, sz, nppcontext));

        // print2D("Values after div", d_div[i], compute_stream);

        Npp32f *const aDst_arr[3] = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};

        NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f *>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch,
                                         aDst_arr, d_channelA[i].ptr().dims.pitch, sz, nppcontext));

        /* print2D("Split X", d_channelA[i], compute_stream);
        print2D("Split Y", d_channelB[i], compute_stream);
        print2D("Split Z", d_channelC[i], compute_stream);*/
      }

      STOP_NPP_START_FK_BENCHMARK

      // do the same via fk
      const auto readOp = fk::resize<fk::PerThreadRead<fk::_2D, uchar3>, float3, fk::InterpolationType::INTER_LINEAR,
                                     BATCH, fk::AspectRatio::IGNORE_AR>(d_crop_fk, fk::Size(UP_W, UP_H), BATCH);
      auto split = fk::Write<fk::TensorSplit<float3>>{d_tensor.ptr()};

      fk::executeOperations(compute_stream, readOp, colorConvert, multiply, sub, div, split);
      STOP_FK_BENCHMARK
      // copy tensor
      gpuErrchk(cudaMemcpyAsync(h_tensor.ptr().data, d_tensor.ptr().data, h_tensor.sizeInBytes(),
                                cudaMemcpyDeviceToHost, compute_stream));

      // Bucle final de copia (NPP)
      for (int i = 0; i < BATCH; ++i) {
        const auto d_dims = d_channelA[i].dims();
        const auto h_dims = h_channelA[i].dims();

        gpuErrchk(cudaMemcpy2DAsync(h_channelA[i].ptr().data, h_dims.pitch, d_channelA[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelB[i].ptr().data, h_dims.pitch, d_channelB[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelC[i].ptr().data, h_dims.pitch, d_channelC[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
      }

      gpuErrchk(cudaStreamSynchronize(compute_stream));

      // free NPP Data
      gpuErrchk(cudaFree(dBatchSrc));
      gpuErrchk(cudaFree(dBatchDst));
      gpuErrchk(cudaFree(dBatchROI));
      gpuErrchk(cudaFreeHost(hBatchSrc));
      gpuErrchk(cudaFreeHost(hBatchDst));
      gpuErrchk(cudaFreeHost(hBatchROI))

          // compare data

      const float TOLERANCE = 1e-3;
      for (int j = 0; j < BATCH; ++j) {
        for (int c = 0; c < 3; ++c) {
          for (int i = 0; i < COLOR_PLANE; ++i) {
            const int i_tensor = i + (COLOR_PLANE * c);
            const float result = h_tensor.ptr().data[(j * IMAGE_STRIDE) + i_tensor];
            switch (c) {
            case RED: {
              float nppResult = h_channelA[j].ptr().data[i];
              float diff = std::abs(result - nppResult);

              passed &= diff < TOLERANCE;

              break;
            }
            case GREEN: {
              float nppResult = h_channelB[j].ptr().data[i];
              float diff = std::abs(result - nppResult);
              passed &= diff < TOLERANCE;
              break;
            }
            case BLUE: {
              float nppResult = h_channelC[j].ptr().data[i];
              float diff = std::abs(result - nppResult);
              passed &= diff < TOLERANCE;
              break;
            }
            }
          }
        }
      }

      if (!passed) {
        std::cout << "BATCH size = " << BATCH << std::endl;
        std::cout << "Results fk:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + h_tensor.dims().plane_pitch);
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + (h_tensor.dims().plane_pitch * 2));
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
        }

        std::cout << "Results npp:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::_2D>::cr_point(fk::Point(x, y), h_channelA[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::_2D>::cr_point(fk::Point(x, y), h_channelB[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::_2D>::cr_point(fk::Point(x, y), h_channelC[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
        }
      }
    }

    catch (const std::exception &e) {
      error_s << e.what();
      passed = false;
      exception = true;
    }

    if (!passed) {
      if (!exception) {
        std::cout << "Test failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
      } else {
        std::cout << "Test failed!! EXCEPTION: " << error_s.str() << std::endl;
      }
    }

    return passed;
  }
  return passed;
}

template <size_t... Is>
bool launch_test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y,
                                           std::index_sequence<Is...> seq, cudaStream_t compute_stream, bool enabled) {
  bool passed = true;
  int dummy[] = {
      (passed &= test_npp_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled),
       0)...};
  (void)dummy;

  return passed;
}

int launch() {
  constexpr size_t NUM_ELEMS_X = 3840;
  constexpr size_t NUM_ELEMS_Y = 2160;

  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&stream));

  // Warmup test execution
  test_npp_batchresize_x_split3D<FIRST_VALUE>(NUM_ELEMS_X, NUM_ELEMS_Y, stream, true);

  std::unordered_map<std::string, bool> results;
  results["test_npp_batchresize_x_split3D"] = true;
  std::make_index_sequence<batchValues.size()> iSeq{};

  results["test_npp_batchresize_x_split3D"] &=
      launch_test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);
  CLOSE_BENCHMARK
  int returnValue = 0;
  for (const auto &[key, passed] : results) {
    if (passed) {
      std::cout << key << " passed!!" << std::endl;
    } else {
      std::cout << key << " failed!!" << std::endl;
      returnValue = -1;
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
  return returnValue;
}