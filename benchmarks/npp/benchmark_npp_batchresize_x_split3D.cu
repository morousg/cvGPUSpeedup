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

#include "tests/testsCommon.cuh"
#include <npp.h>

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
	const char* file,
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

NppStreamContext initNppStreamContext(const cudaStream_t& stream);
NppStreamContext initNppStreamContext(const cudaStream_t& stream)
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
	NppStreamContext nppstream = { stream, device, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
								  prop.maxThreadsPerBlock, prop.sharedMemPerBlock, ccmajor, ccminor, flags };
	return nppstream;
}

template <int BATCH>
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cudaStream_t& compute_stream, bool enabled)
{
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled)
	{
		 
		try
		{
			const float alpha = 0.3f;
			const uchar CROP_W = 60;
			const uchar CROP_H = 120;
			const uchar UP_W = 64;
			const uchar UP_H = 128;

			std::array<fk::Ptr2D<uchar3>, BATCH> d_input;
			std::array<fk::Ptr2D<float3>, BATCH> d_crop, d_mul, d_sub, d_div, d_inputf;
			std::array<fk::Ptr2D<float>, BATCH> channelA, channelB, channelC;
			std::array<fk::Ptr2D<float>, BATCH> hchannelA, hchannelB, hchannelC;

			NppiImageDescriptor* hBatchSrc = nullptr, * dBatchSrc = nullptr, * hBatchDst = nullptr, * dBatchDst = nullptr;
			NppiResizeBatchROI_Advanced* dBatchROI = nullptr, * hBatchROI = nullptr;

			// init data adn set to inital value
			NppStreamContext nppcontext = initNppStreamContext(compute_stream);

			// source images (rgb 8bit)
			const Npp8u aValue[3] = { 5, 5, 5 };
			const Npp32f fValue[3] = { 1.0f, 3.0f, 10.0f };
			const Npp32f mulValue[3] = { alpha, alpha, alpha };
			const Npp32f subValue[3] = { 1.f, 4.f, 3.2f };
			const Npp32f divValue[3] = { 1.f, 4.f, 3.2f };
			const NppiSize sz{ UP_W, UP_H };
			const NppiSize szcrop = { CROP_W, CROP_H };
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

			for (int i = 0; i < BATCH; ++i)
			{
				d_input[i] = fk::Ptr2D<uchar3>(UP_W, UP_H);
				d_inputf[i] = fk::Ptr2D<float3>(UP_W, UP_H);
				NPP_CHECK(nppiSet_8u_C3R_Ctx(aValue, reinterpret_cast<Npp8u*>(d_input[i].ptr().data), d_input[i].dims().pitch, sz, nppcontext));
				hBatchSrc[i].pData = d_inputf[i].ptr().data;
				hBatchSrc[i].nStep = d_inputf[i].ptr().dims.pitch;
				hBatchSrc[i].oSize = sz;
			}
			std::vector <std::array<Npp32f*, 3>> aDst;
			// dest images (Rgb, 32f)
			for (int i = 0; i < BATCH; ++i)
			{
				d_crop[i] = fk::Ptr2D<float3>(i + CROP_W, i + CROP_H);
				d_mul[i] = fk::Ptr2D<float3>(i + CROP_W, i + CROP_H);
				d_sub[i] = fk::Ptr2D<float3>(i + CROP_W, i + CROP_H);
				d_div[i] = fk::Ptr2D<float3>(i + CROP_W, i + CROP_H);

				hBatchDst[i].pData = d_crop[i].ptr().data;
				hBatchDst[i].nStep = d_crop[i].ptr().dims.pitch;
				hBatchDst[i].oSize = szcrop;

				NPP_CHECK(nppiSet_32f_C3R_Ctx(fValue, reinterpret_cast<Npp32f*>(d_crop[i].ptr().data), d_crop[i].dims().pitch, sz, nppcontext));
				NPP_CHECK(nppiSet_32f_C3R_Ctx(fValue, reinterpret_cast<Npp32f*>(d_mul[i].ptr().data), d_mul[i].dims().pitch, sz, nppcontext));
				NPP_CHECK(nppiSet_32f_C3R_Ctx(fValue, reinterpret_cast<Npp32f*>(d_sub[i].ptr().data), d_sub[i].dims().pitch, sz, nppcontext));
				NPP_CHECK(nppiSet_32f_C3R_Ctx(fValue, reinterpret_cast<Npp32f*>(d_div[i].ptr().data), d_div[i].dims().pitch, sz, nppcontext));
				// Bucle inicial d'alocatacio
				channelA[i] = fk::Ptr2D<float>(CROP_W, CROP_H);
				channelB[i] = fk::Ptr2D<float>(CROP_W, CROP_H);
				channelC[i] = fk::Ptr2D<float>(CROP_W, CROP_H);
				hchannelA[i] = fk::Ptr2D<float>(CROP_W, CROP_H, channelA[i].ptr().dims.pitch, fk::MemType::HostPinned);
				hchannelB[i] = fk::Ptr2D<float>(CROP_W, CROP_H, channelB[i].ptr().dims.pitch, fk::MemType::HostPinned);
				hchannelC[i] = fk::Ptr2D<float>(CROP_W, CROP_H, channelC[i].ptr().dims.pitch, fk::MemType::HostPinned);
				std::array<Npp32f*, 3> ptrs = { reinterpret_cast<Npp32f*>(channelA[i].ptr().data),
				reinterpret_cast<Npp32f*>(channelB[i].ptr().data),
				reinterpret_cast<Npp32f*>(channelC[i].ptr().data) };
				aDst.push_back(ptrs);

			}

			// ROI
			for (int i = 0; i < BATCH; ++i)
			{
				NppiRect srcrect = { i, i, CROP_W, CROP_H };
				NppiRect dstrect = { 0, 0, CROP_W, CROP_H };
				hBatchROI[i].oSrcRectROI = srcrect;
				hBatchROI[i].oDstRectROI = dstrect;
			}
			 
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, compute_stream));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, compute_stream));
			gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (dBatchROI), hBatchROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, compute_stream));

			// NPP version
			// convert to 32f			

			for (int i = 0; i < BATCH; ++i)
			{
				NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<Npp8u*>(d_input[i].ptr().data),
					d_input[i].ptr().dims.pitch,
					reinterpret_cast<Npp32f*>(d_inputf[i].ptr().data), d_inputf[i].ptr().dims.pitch, hBatchSrc[i].oSize, nppcontext));
			}


			NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(CROP_W, CROP_H, dBatchSrc, dBatchDst, dBatchROI, BATCH, NPPI_INTER_LINEAR, nppcontext));
			// crop array of images using batch resize+ ROIs

			// asume RGB->BGR
			const int aDstOrder[3] = { 2, 1, 0 };
			for (int i = 0; i < BATCH; ++i)
			{
				NPP_CHECK(nppiSwapChannels_32f_C3IR_Ctx(reinterpret_cast<Npp32f*>(d_crop[i].ptr().data), d_crop[i].ptr().dims.pitch, sz, aDstOrder, nppcontext));
				//	gpuErrchk(cudaStreamSynchronize(compute_stream));
	//				NPP_CHECK(nppiSwapChannels_32f_C3IR_Ctx(reinterpret_cast<Npp32f*>(dBatchDst[i].pData), dBatchDst[i].nStep, sz, aDstOrder, nppcontext));


				NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_crop[i].ptr().data), d_crop[i].ptr().dims.pitch, mulValue,
					reinterpret_cast<Npp32f*>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch, szcrop, nppcontext));

				NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch, subValue,
					reinterpret_cast<Npp32f*>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch, szcrop, nppcontext));

				NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch, divValue,
					reinterpret_cast<Npp32f*>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch, szcrop, nppcontext));

				//split
				 
				NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f*>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch,
					reinterpret_cast<Npp32f**>(aDst[i].front()), channelA[i].ptr().dims.pitch, szcrop, nppcontext));
			}
			 
			//Bucle final de copia
			for (int i = 0; i < BATCH; ++i)
			{
				gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (hchannelA[i].ptr().data), channelA[i].ptr().data, sizeof(float) * CROP_W * CROP_H,
					cudaMemcpyHostToDevice, compute_stream));
				gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (hchannelB[i].ptr().data), channelB[i].ptr().data, sizeof(float) * CROP_W * CROP_H,
					cudaMemcpyHostToDevice, compute_stream));
				gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**> (hchannelC[i].ptr().data), channelC[i].ptr().data, sizeof(float) * CROP_W * CROP_H,
					cudaMemcpyHostToDevice, compute_stream));

			} 

			gpuErrchk(cudaStreamSynchronize(compute_stream));
			gpuErrchk(cudaFree(dBatchSrc));
			gpuErrchk(cudaFree(dBatchDst));
			gpuErrchk(cudaFree(dBatchROI));
			gpuErrchk(cudaFreeHost(hBatchSrc));
			gpuErrchk(cudaFreeHost(hBatchDst));
			gpuErrchk(cudaFreeHost(hBatchROI))

				//do the same via fk
/*
			std::array<fk::Ptr2D<uchar3>, BATCH> d_crop1;
			for (int i = 0; i < BATCH; i++) {
				d_crop1[i] = d_input[i].crop2D(fk::Point(i, i, 0), fk::PtrDims<fk::_2D>(i + 60, i + 120, 0));

			} 
 
	 

		const auto readOp =
				fk::resize<fk::PerThreadRead<fk::_2D, uchar3>, 
				uchar3, 
				fk::InterpolationType::INTER_LINEAR, 
				BATCH, 
				fk::AspectRatio::IGNORE_AR>(d_crop1, fk::Size(CROP_W, CROP_W), BATCH);
			auto colorConvert = fk::Unary<fk::VectorReorder<float3, 2, 1, 0>>{};
			auto multiply = fk::Binary<fk::Mul<float3>>{ fk::make_set<float3>(4.f) };
			auto sub = fk::Binary<fk::Sub<float3>>{ fk::make_set<float3>(4.f) };
			auto div = fk::Binary<fk::Div<float3>>{ fk::make_set<float3>(4.f) };

  
			fk::SplitWriteParams<fk::_2D, float3> writeParams;
			writeParams.x = fk::Ptr2D<float> (CROP_W, CROP_H);// hA.ptr().data;/*RawPtr<_2D, float>;
			writeParams.y = fk::Ptr2D<float>(CROP_W, CROP_H);// hB.ptr().data;/*RawPtr<_2D, float>;
			writeParams.z = fk::Ptr2D<float>(CROP_W, CROP_H); //hC.ptr().data;/*RawPtr<_2D, float>;

			auto split = fk::Write<fk::SplitWrite<fk::_2D, float3>>{ writeParams };

			fk::executeOperations(compute_stream, readOp, colorConvert, multiply, sub, div, split);
			*/
		}

		catch (const std::exception& e)
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
template <size_t... Is>
bool test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cudaStream_t compute_stream, bool enabled)
{
	bool passed = true;
	int dummy[] = { (passed &= test_npp_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled), 0)... };
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

	results["test_npp_batchresize_x_split3D"] &= test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);

	int returnValue = 0;
	for (const auto& [key, passed] : results)
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