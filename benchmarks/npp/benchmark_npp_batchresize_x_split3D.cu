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

#include <sstream>
#include <numeric>
#include "tests/testsCommon.cuh"
#include <npp.h>

#include "tests/testUtils.cuh"
#include <nppi_geometry_transforms.h>
#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/algorithms/image_processing/resize_builders.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <cvGPUSpeedup.cuh>

#include "tests/main.h"

constexpr size_t NUM_EXPERIMENTS = 9;
constexpr size_t FIRST_VALUE = 10;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;
constexpr inline void nppAssert(NppStatus code,
								const char *file,
								int line,
								bool abort = true)
{
	if (code != NPP_SUCCESS)
	{
		std::cout << "NPP failure: " << " File: " << file << " Line:" << line << std::endl;
		if (abort)
			throw std::exception();
	}
}

#define NPP_CHECK(ans)                              \
	{                                               \
		nppAssert((ans), __FILE__, __LINE__, true); \
	}

NppStreamContext initNppStreamContext(const cudaStream_t &stream);
NppStreamContext initNppStreamContext(const cudaStream_t &stream)
{
	int device = 0;
	int ccmajor = 0;
	int ccminor = 0;
	uint flags;

	cudaDeviceProp prop;
	gpuErrchk(cudaGetDevice(&device))
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
	gpuErrchk(cudaDeviceGetAttribute(&ccmajor, cudaDevAttrComputeCapabilityMinor, device));
	gpuErrchk(cudaDeviceGetAttribute(&ccminor, cudaDevAttrComputeCapabilityMajor, device));
	gpuErrchk(cudaStreamGetFlags(stream, &flags));
	NppStreamContext nppstream = {stream, device, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
								  prop.maxThreadsPerBlock, prop.sharedMemPerBlock, ccmajor, ccminor, flags};
	return nppstream;
}

template <int BATCH>
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cudaStream_t &compute_stream, bool enabled)
{
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled)
	{
		const float alpha = 0.3f;
		const uchar CROP_W = 2;
		const uchar CROP_H = 2;
		const uchar UP_W = 4;
		const uchar UP_H = 4;
		try
		{
			constexpr uchar3 init_val{1,2,3};
			fk::Ptr2D<uchar3> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
			fk::setTo(init_val, d_input, compute_stream);
			fk::Ptr2D<float3> d_input_f(NUM_ELEMS_X, NUM_ELEMS_Y);
			std::array<fk::Ptr2D<float3>, BATCH> d_resized_npp, d_mul, d_sub, d_div;
			std::array<fk::Ptr2D<float>, BATCH> channelA, channelB, channelC;
			std::array<fk::Ptr2D<float>, BATCH> hchannelA, hchannelB, hchannelC;
			fk::Tensor<float> d_tensornpp(UP_W, UP_H, BATCH, 3);
			NppiImageDescriptor* hBatchSrc = nullptr, * dBatchSrc = nullptr, * hBatchDst = nullptr, * dBatchDst = nullptr;
			NppiResizeBatchROI_Advanced* dBatchROI = nullptr, * hBatchROI = nullptr;

			// init data adn set to inital value
			NppStreamContext nppcontext = initNppStreamContext(compute_stream);

			// source images (rgb 8bit)
			const Npp8u aValue[3] = { init_val.x ,init_val.y ,init_val.z };
			const Npp32f mulValue[3] = { alpha, alpha, alpha };
			const Npp32f subValue[3] =  { 1.f, 4.f, 3.2f };
			const Npp32f divValue[3] =  { 1.f, 4.f, 3.2f };
			const NppiSize sz{ UP_W, UP_H };
			const NppiSize szcrop = { CROP_W, CROP_H };
			// asume RGB->BGR
			const int aDstOrder[3] = { 2, 1, 0 };
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

				//printTensor(d_input_tensor);
			
			for (int i = 0; i < BATCH; ++i) {
				//NPP_CHECK(nppiSet_8u_C3R_Ctx(aValue, reinterpret_cast<Npp8u*>(d_input[i].ptr().data), d_input[i].dims().pitch, sz, nppcontext));
				//gpuErrchk(cudaMemsetAsync(reinterpret_cast<Npp8u*>(d_input[i].ptr().data), aValue[0], sizeof(uchar3) * UP_W * UP_H,compute_stream));
			
				hBatchSrc[i].pData = d_input_f.ptr().data;
				hBatchSrc[i].nStep = d_input_f.ptr().dims.pitch;
				hBatchSrc[i].oSize = sz;
			}
			
			std::vector<std::array<Npp32f*, 3>> aDst;
			// dest images (Rgb, 32f)
			for (int i = 0; i < BATCH; ++i)
			{
				d_resized_npp[i] = fk::Ptr2D<float3>(UP_W, UP_H);
				d_mul[i] = fk::Ptr2D<float3>(UP_W, UP_H);
				d_sub[i] = fk::Ptr2D<float3>(UP_W, UP_H);
				d_div[i] = fk::Ptr2D<float3>(UP_W, UP_H);

				hBatchDst[i].pData = d_resized_npp[i].ptr().data;
				hBatchDst[i].nStep = d_resized_npp[i].ptr().dims.pitch;
				hBatchDst[i].oSize = sz;

				channelA[i] = fk::Ptr2D<float>(UP_W, UP_H);
				channelB[i] = fk::Ptr2D<float>(UP_W, UP_H);
				channelC[i] = fk::Ptr2D<float>(UP_W, UP_H);

				hchannelA[i] = fk::Ptr2D<float>(UP_W, UP_H, channelA[i].ptr().dims.pitch, fk::MemType::HostPinned);
				hchannelB[i] = fk::Ptr2D<float>(UP_W, UP_H, channelB[i].ptr().dims.pitch, fk::MemType::HostPinned);
				hchannelC[i] = fk::Ptr2D<float>(UP_W, UP_H, channelC[i].ptr().dims.pitch, fk::MemType::HostPinned);

				std::array<Npp32f*, 3> ptrs = { reinterpret_cast<Npp32f*>(channelA[i].ptr().data),
												reinterpret_cast<Npp32f*>(channelB[i].ptr().data),
												reinterpret_cast<Npp32f*>(channelC[i].ptr().data) };
				aDst.push_back(ptrs);
			}

			// ROI
			for (int i = 0; i < BATCH; ++i)
			{
				NppiRect srcrect = { i, i, CROP_W, CROP_H };
				NppiRect dstrect = { 0, 0, UP_W, UP_H };
				hBatchROI[i].oSrcRectROI = srcrect;
				hBatchROI[i].oDstRectROI = dstrect;
			}

			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, compute_stream));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, compute_stream));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchROI), hBatchROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, compute_stream));

			// NPP version
			// convert to 32f
			
			NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<Npp8u*>(d_input.ptr().data),
				d_input.ptr().dims.pitch,
				reinterpret_cast<Npp32f*>(d_input_f.ptr().data), d_input_f.ptr().dims.pitch,
				NppiSize{static_cast<int>(NUM_ELEMS_X), static_cast<int>(NUM_ELEMS_Y)}, nppcontext));

 
			NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dBatchSrc, dBatchDst, dBatchROI, BATCH, NPPI_INTER_LINEAR, nppcontext));
			// crop array of images using batch resize+ ROIs
			//force fill 5
			constexpr int COLOR_PLANE = UP_H * UP_W;
			constexpr int IMAGE_STRIDE = (COLOR_PLANE * 3);
			
 

			for (int i = 0; i < BATCH; ++i)
			{
				 
				NPP_CHECK(nppiSwapChannels_32f_C3IR_Ctx(reinterpret_cast<Npp32f*>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch, sz, aDstOrder, nppcontext));
				
				NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch, mulValue,
					reinterpret_cast<Npp32f*>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch, sz, nppcontext));

				NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch, subValue,
				reinterpret_cast<Npp32f*>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch, sz, nppcontext));

		NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch, divValue,
					reinterpret_cast<Npp32f*>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch, sz, nppcontext));

				const auto vals = aDst[i].front();
				Npp32f* const aDst1[3] = { &vals[0], &vals[1], &vals[2] };

				 NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f*>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch,
			 		aDst1, channelA[i].ptr().dims.pitch, sz, nppcontext));
			}
			//for (int i = 0; i < BATCH; ++i)
			//{
			//	fk::setTo(fk::make_set<float3>(5.0f), d_div[i]);
				//fk::setTo(fk::make_set<float>(5.0f), channelA[i]);
				//fk::setTo(fk::make_set<float>(5.0f), channelB[i]);
				//fk::setTo(fk::make_set<float>(5.0f), channelC[i]);
			//}
			// Bucle final de copia
			for (int i = 0; i < BATCH; ++i)
			{
				gpuErrchk(cudaMemcpy2DAsync(reinterpret_cast<void**>(hchannelA[i].ptr().data), hchannelA[i].dims().pitch, 
					channelA[i].ptr().data, channelA[i].dims().pitch, channelA[i].dims().width * sizeof(float), CROP_H,
					cudaMemcpyDeviceToHost, compute_stream));
				gpuErrchk(cudaMemcpy2DAsync(reinterpret_cast<void**>(hchannelB[i].ptr().data), hchannelB[i].dims().pitch,
					channelB[i].ptr().data, channelB[i].dims().pitch, channelB[i].dims().width * sizeof(float), CROP_H,
					cudaMemcpyDeviceToHost, compute_stream));
				gpuErrchk(cudaMemcpy2DAsync(reinterpret_cast<void**>(hchannelC[i].ptr().data), hchannelC[i].dims().pitch,
					channelC[i].ptr().data, channelC[i].dims().pitch, channelC[i].dims().width * sizeof(float), CROP_H,
					cudaMemcpyDeviceToHost, compute_stream));
			}

			gpuErrchk(cudaStreamSynchronize(compute_stream));
			gpuErrchk(cudaFree(dBatchSrc));
			gpuErrchk(cudaFree(dBatchDst));
			gpuErrchk(cudaFree(dBatchROI));
			gpuErrchk(cudaFreeHost(hBatchSrc));
			gpuErrchk(cudaFreeHost(hBatchDst));
			gpuErrchk(cudaFreeHost(hBatchROI))

				fk::Tensor<float>
				h_tensor(UP_W, UP_H, BATCH, 3, fk::MemType::HostPinned);
			fk::Tensor<float> d_tensor(UP_W, UP_H, BATCH, 3);

			// do the same via fk

			std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> d_crop_fk;
			for (int i = 0; i < BATCH; i++)
			{
				d_crop_fk[i] = d_input.crop2D(fk::Point(i, i), fk::PtrDims<fk::_2D>(60, 120));
			}

			const auto readOp =
				fk::resize<fk::PerThreadRead<fk::_2D, uchar3>,
				float3,
				fk::InterpolationType::INTER_LINEAR,
				BATCH,
				fk::AspectRatio::IGNORE_AR>(d_crop_fk, fk::Size(UP_W, UP_W), BATCH);
			auto colorConvert = fk::Unary<fk::VectorReorder<float3, 2, 1, 0>>{};
			auto multiply = fk::Binary<fk::Mul<float3>>{ fk::make_<float3>(mulValue[0], mulValue[1], mulValue[2]) };
			auto sub = fk::Binary<fk::Sub<float3>>{ fk::make_<float3>(subValue[0], subValue[1], subValue[2]) };
			auto div = fk::Binary<fk::Div<float3>>{ fk::make_<float3>(divValue[0], divValue[1], divValue[2]) };
			auto split = fk::Write<fk::TensorSplit<float3>>{ d_tensor.ptr() };

			fk::executeOperations(compute_stream, readOp, colorConvert, multiply, sub, div, split);

			// copy tensor

			gpuErrchk(cudaMemcpy2DAsync(h_tensor.ptr().data, h_tensor.dims().pitch, d_tensor.ptr().data, d_tensor.dims().pitch,
				d_tensor.dims().width * sizeof(float) * 3, d_tensor.dims().height, cudaMemcpyDeviceToHost, compute_stream));

			gpuErrchk(cudaStreamSynchronize(compute_stream));

			// compare data


			const float TOLERANCE = 1e-3;
			for (int j = 0; j < BATCH; ++j)
			{
				for (int c = 0; c < 3; ++c)
				{
					for (int i = 0; i < COLOR_PLANE; ++i)
					{
						const int i_tensor = i + (COLOR_PLANE * c);
						const float result = h_tensor.ptr().data[(j * IMAGE_STRIDE) + i_tensor];
						switch (c)
						{
						case 0:
						{
							float nppResult = hchannelA[j].ptr().data[i];
							float diff = std::abs(result - nppResult);

							passed &= diff < TOLERANCE;

							break;
						}
						case 1:
						{
							float nppResult = hchannelB[j].ptr().data[i];
							float diff = std::abs(result - nppResult);
							passed &= diff < TOLERANCE;
							break;
						}
						case 2:
						{
							float nppResult = hchannelC[j].ptr().data[i];
							float diff = std::abs(result - nppResult);
							passed &= diff < TOLERANCE;
							break;
						}
						}
					}
				}
			}



			if (!passed)
			{
				std::cout << "fk:" << std::endl;
				printTensor(h_tensor);

				std::cout << "npp:" << std::endl;
				for (int b = 0; b < BATCH; ++b)
				{

					for (int i = 0; i < COLOR_PLANE; ++i)
					{
						std::cout << hchannelA[b].ptr().data[i] << " | ";
						std::cout << hchannelB[b].ptr().data[i] << " | ";
						std::cout << hchannelC[b].ptr().data[i] << " | "<<std::endl<<"..................." << std::endl ;

					}
				}
			}
		}

		catch (const std::exception &e)
		{
			error_s << e.what();
			passed = false;
			exception = true;
		}

		if (!passed)
		{

			if (!exception)
			{
				std::stringstream ss;
				//			ss << "test_npp_batchresize_x_split3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
			}
			else
			{
				std::stringstream ss;
				//	ss << "test_npp_batchresize_x_split3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
			}
		}

		return passed;
	}
	return passed;
}

template <size_t First, size_t... Rest>
bool launch_test_npp_batchresize_x_split3D_helper(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<First, Rest...> seq, cudaStream_t compute_stream, bool enabled)
{
	if constexpr (sizeof...(Rest) == 0)
	{
		return test_npp_batchresize_x_split3D<batchValues[First]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled);
	}
	else
	{
		size_t currentBatch = First;
		const bool passed1 = test_npp_batchresize_x_split3D<batchValues[First]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled);
		const bool passed2 = launch_test_npp_batchresize_x_split3D_helper<Rest...>(NUM_ELEMS_X, NUM_ELEMS_Y, std::index_sequence<Rest...>{}, compute_stream, enabled);
		return passed1 && passed2;
	}
}

template <size_t... Is>
bool launch_test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cudaStream_t compute_stream, bool enabled)
{
	/*bool passed = true;
	int dummy[] = {(passed &= test_npp_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled), 0)...};
	(void)dummy;*/

	const bool passed = launch_test_npp_batchresize_x_split3D_helper<Is...>(NUM_ELEMS_X, NUM_ELEMS_Y, seq, compute_stream, enabled);

	return passed;
}

int launch()
{
	constexpr size_t NUM_ELEMS_X = 3840;
	constexpr size_t NUM_ELEMS_Y = 2160;

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	std::unordered_map<std::string, bool> results;
	results["test_npp_batchresize_x_split3D"] = true;
	std::make_index_sequence<batchValues.size()> iSeq{};

	results["test_npp_batchresize_x_split3D"] &= launch_test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);

	int returnValue = 0;
	for (const auto &[key, passed] : results)
	{
		if (passed)
		{
			std::cout << key << " passed!!" << std::endl;
		}
		else
		{
			std::cout << key << " failed!!" << std::endl;
			returnValue = -1;
		}
	}
	gpuErrchk(cudaStreamDestroy(stream));
	return returnValue;
}