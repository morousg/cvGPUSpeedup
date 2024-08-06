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
#include <cvGPUSpeedup.cuh>

#include "tests/main.h"

constexpr char VARIABLE_DIMENSION[]{ "Batch size" };
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
		const float alpha = 0.3f;
		const uchar3 val_init = { 5u, 5u, 5u };
		const float3 val_alpha = { alpha, alpha, alpha };
		const float3 val_sub = { 1.f, 4.f, 3.2f };
		const float3 val_div = { 1.f, 4.f, 3.2f };
		const uchar CROP_W = 60;
		const uchar CROP_H = 120;
		const uchar UP_W = 64;
		const uchar UP_H = 128;
		try
		{

			std::array<fk::Ptr2D<uchar3>, BATCH> d_input;
			std::array<fk::Ptr2D<uchar3>, BATCH> crops_2d;
			std::array<fk::Ptr2D<float3>, BATCH> d_output;
			std::array<NppiImageDescriptor, BATCH> srccropsdesc;
			//init data
			for (int i = 0; i < BATCH; ++i)
			{
				fk::Ptr2D<uchar3> h_temp(UP_W, UP_H, 0, fk::MemType::HostPinned);
				for (uint y = 0; y < UP_H; y++) {
					for (uint x = 0; x < UP_W; x++) {
						const fk::Point p{ x, y, 0 };
						*fk::PtrAccessor<fk::_2D>::point(p, h_temp.ptr()) = val_init;
					}
				}

				fk::Ptr2D<uchar3> temp(UP_W, UP_H);
				d_input[i] = temp;
				gpuErrchk(cudaMemcpy2DAsync(temp.ptr().data, temp.dims().pitch, h_temp.ptr().data, h_temp.dims().pitch,
					h_temp.dims().width * sizeof(uchar3), h_temp.dims().height, cudaMemcpyHostToDevice, compute_stream));
			}


			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				const fk::Ptr2D<uchar3> d_cropped = d_input[crop_i].crop2D(fk::Point(crop_i, crop_i), fk::PtrDims<fk::_2D>{CROP_W, CROP_H, d_input[crop_i].ptr().dims.pitch});

				crops_2d[crop_i] = d_cropped;
			}

			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{

				srccropsdesc[crop_i].pData = d_input[crop_i].ptr().data;
				srccropsdesc[crop_i].nStep = d_input[crop_i].ptr().dims.pitch;
				srccropsdesc[crop_i].oSize = { crop_i + CROP_W, crop_i + CROP_H };

			}


			const fk::Size up(UP_W, UP_H);
			fk::Ptr2D<uchar3> d_up((int)NUM_ELEMS_X, (int)NUM_ELEMS_Y);;
			fk::Ptr2D<float3> d_temp((int)NUM_ELEMS_X, (int)NUM_ELEMS_Y);;
			fk::Ptr2D<float3> d_temp2((int)NUM_ELEMS_X, (int)NUM_ELEMS_Y);

			std::array<fk::Ptr2D<float3>, BATCH> d_output_crop_npp;
			std::array<std::vector<fk::Ptr2D<float3>>, BATCH> d_output_cvGS;

			fk::Tensor<uchar3> d_tensor_output;
			std::array<fk::Ptr2D<float3>, BATCH> crops;

			std::array<NppiImageDescriptor, BATCH> dstcropsdesc;
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{

				dstcropsdesc[crop_i].pData = d_output_crop_npp[crop_i].ptr().data;
				dstcropsdesc[crop_i].nStep = d_output_crop_npp[crop_i].ptr().dims.pitch;
				dstcropsdesc[crop_i].oSize = { up.width, up.height };
			}

			// NPP version

			NppStreamContext nppcontext = initNppStreamContext(compute_stream);

			for (int i = 0; i < BATCH; ++i)
			{
				NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<Npp8u*>(d_input[i].ptr().data),
					d_input[i].ptr().dims.pitch,
					reinterpret_cast<Npp32f*>(d_output[i].ptr().data), d_output[i].ptr().dims.pitch, srccropsdesc[i].oSize, nppcontext));
			}
			//(for i=0;i<BATCH;++i)
			// nppiConvert_8u32f_AC3
			//
			// nppiResizeBatch_32f_C3R_Ctx()

			// //(for i=0;i<BATCH;++i)
			// nppiSwapChannels_32f_C3IR_Ctx()
			// nppiMulC_32f_C3IRSfs_Ctx()
			// nppiSubC_32f_C3RSfs_Ctx/()
			// nppiDivC_32f_C3RSfs_Ctx()
			// split???? nppiCopy_32f_C3P3R_Ctx


		// cvGPUSpeedup
			/*cvGS::executeOperations(compute_stream,
				cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
				cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_TYPE_O>(),
				cvGS::multiply<CV_TYPE_O>(val_alpha),
				cvGS::subtract<CV_TYPE_O>(val_sub),
				cvGS::divide<CV_TYPE_O>(val_div),
				cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				*/
				//			d_tensor_output.download(h_tensor_output, compute_stream);

		//					// Verify results
			//for (int crop_i = 0; crop_i < BATCH; crop_i++)
		//	{
			//	for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++)
				//{
					//					d_output_cv[crop_i].at(i).download(h_cvResults[crop_i].at(i), compute_stream);
			//	}
		//	}
			gpuErrchk(cudaStreamSynchronize(compute_stream));

			/*
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				cv::Mat row = h_tensor_output.row(crop_i);
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++)
				{
					int planeStart = i * up.width * up.height;
					int planeEnd = ((i + 1) * up.width * up.height) - 1;
					cv::Mat plane = row.colRange(planeStart, planeEnd);
					h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
				}
			}

			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++)
				{
					cv::Mat cvRes = h_cvResults[crop_i].at(i);
					cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
					diff = cv::abs(cvRes - cvGSRes);
					bool passedThisTime = checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
					passed &= passedThisTime;
				}
			}*/
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