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
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <npp.h>

#include "tests/main.h"
#include <nppdefs.h>

constexpr char VARIABLE_DIMENSION[]{"Batch size"};
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

#define nppAssert(ans)                              \
	{                                               \
		nppAssert((ans), __FILE__, __LINE__, true); \
	}

NppStreamContext initNppStream(cudaStream_t stream);

NppStreamContext initNppStream(cudaStream_t stream)
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
	NppStreamContext nppstream ={stream, device, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, 
		prop.maxThreadsPerBlock, prop.sharedMemPerBlock, ccmajor, ccminor, flags};
	return nppstream;
}

template <const uint NPP_INPUT, const uint NPP_OUTPUT, const uint NPP_CHANNELS, const uint BATCH>
bool test_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, const cudaStream_t &stream, bool enabled)
{
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled)
	{
		struct Parameters
		{

			cv::Scalar init;
			cv::Scalar alpha;
			cv::Scalar val_sub;
			cv::Scalar val_div;
		};

		double alpha = 0.3;

		std::vector<Parameters> params = {
			{{2u}, {alpha}, {1.f}, {3.2f}},
			{{2u, 37u}, {alpha, alpha}, {1.f, 4.f}, {3.2f, 0.6f}},
			{{5u, 5u, 5u}, {alpha, alpha, alpha}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f}},
			{{2u, 37u, 128u, 20u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}};

		cv::Scalar val_init = params.at(CV_MAT_CN(NPP_OUTPUT) - 1).init;
		cv::Scalar val_alpha = params.at(CV_MAT_CN(NPP_OUTPUT) - 1).alpha;
		cv::Scalar val_sub = params.at(CV_MAT_CN(NPP_OUTPUT) - 1).val_sub;
		cv::Scalar val_div = params.at(CV_MAT_CN(NPP_OUTPUT) - 1).val_div;

		try
		{

			// pSrc, pSum, pDeviceBuffer are all device pointers.
			NppStreamContext nppstream = initNppStream(stream);
			const uint32_t nLength = NUM_ELEMS_X * NUM_ELEMS_Y;

			std::array<NppiRect, BATCH> crops_2d;
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				NppiRect r{crop_i, crop_j, crop_i + 60, crop_i + 120};
				crops_2d[crop_i] = r;
			}
			NppiSize up={ 64,128 };
			

			cv::cuda::GpuMat d_up(up, NPP_INPUT);
			cv::cuda::GpuMat d_temp(up, NPP_OUTPUT);
			cv::cuda::GpuMat d_temp2(up, NPP_OUTPUT);

			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cv;
			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cvGS;
			std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
			std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

			cv::cuda::GpuMat d_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(NPP_OUTPUT), CV_MAT_DEPTH(NPP_OUTPUT));
			d_tensor_output.step = up.width * up.height * CV_MAT_CN(NPP_OUTPUT) * sizeof(BASE_CUDA_T(NPP_OUTPUT));

			cv::Mat diff(up, CV_MAT_DEPTH(NPP_OUTPUT));
			cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(NPP_OUTPUT), CV_MAT_DEPTH(NPP_OUTPUT));

			std::array<cv::cuda::GpuMat, BATCH> crops;
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				crops[crop_i] = d_input(crops_2d[crop_i]);
				for (int i = 0; i < CV_MAT_CN(NPP_INPUT); i++)
				{
					d_output_cv.at(crop_i).emplace_back(up, CV_MAT_DEPTH(NPP_OUTPUT));
					h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(NPP_OUTPUT));
				}
			}

			constexpr bool correctDept = CV_MAT_DEPTH(NPP_OUTPUT) == CV_32F;

			START_OCV_BENCHMARK
			// OpenCV version
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				// NppStatus nppiGetResizeRect(NppiRect oSrcROI, NppiRect* pDstRect, double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation)//
				cv::cuda::resize(crops[crop_i], d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
				d_up.convertTo(d_temp, NPP_OUTPUT, alpha, cv_stream);
				if constexpr (CV_MAT_CN(NPP_INPUT) == 3 && correctDept)
				{
					cv::cuda::cvtColor(d_temp, d_temp, cv::COLOR_RGB2BGR, 0, cv_stream);
				}
				else if constexpr (CV_MAT_CN(NPP_INPUT) == 4 && correctDept)
				{
					cv::cuda::cvtColor(d_temp, d_temp, cv::COLOR_RGBA2BGRA, 0, cv_stream);
				}
				cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
				cv::cuda::divide(d_temp2, val_div, d_temp, 1.0, -1, cv_stream);
				cv::cuda::split(d_temp, d_output_cv[crop_i], cv_stream);
			}
			STOP_OCV_START_CVGS_BENCHMARK
			// cvGPUSpeedup
			if constexpr (CV_MAT_CN(NPP_INPUT) == 3 && correctDept)
			{
				cvGS::executeOperations(cv_stream,
										cvGS::resize<NPP_INPUT, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
										cvGS::cvtColor<cv::COLOR_RGB2BGR, NPP_OUTPUT>(),
										cvGS::multiply<NPP_OUTPUT>(val_alpha),
										cvGS::subtract<NPP_OUTPUT>(val_sub),
										cvGS::divide<NPP_OUTPUT>(val_div),
										cvGS::split<NPP_OUTPUT>(d_tensor_output, up));
			}
			else if constexpr (CV_MAT_CN(NPP_INPUT) == 4 && correctDept)
			{
				cvGS::executeOperations(cv_stream,
										cvGS::resize<NPP_INPUT, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
										cvGS::cvtColor<cv::COLOR_RGBA2BGRA, NPP_OUTPUT>(),
										cvGS::multiply<NPP_OUTPUT>(val_alpha),
										cvGS::subtract<NPP_OUTPUT>(val_sub),
										cvGS::divide<NPP_OUTPUT>(val_div),
										cvGS::split<NPP_OUTPUT>(d_tensor_output, up));
			}
			else
			{
				cvGS::executeOperations(cv_stream,
										cvGS::resize<NPP_INPUT, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
										cvGS::multiply<NPP_OUTPUT>(val_alpha),
										cvGS::subtract<NPP_OUTPUT>(val_sub),
										cvGS::divide<NPP_OUTPUT>(val_div),
										cvGS::split<NPP_OUTPUT>(d_tensor_output, up));
			}
			STOP_CVGS_BENCHMARK
			d_tensor_output.download(h_tensor_output, cv_stream);

			// Verify results
			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				for (int i = 0; i < CV_MAT_CN(NPP_OUTPUT); i++)
				{
					d_output_cv[crop_i].at(i).download(h_cvResults[crop_i].at(i), cv_stream);
				}
			}

			cv_stream.waitForCompletion();

			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				cv::Mat row = h_tensor_output.row(crop_i);
				for (int i = 0; i < CV_MAT_CN(NPP_OUTPUT); i++)
				{
					int planeStart = i * up.width * up.height;
					int planeEnd = ((i + 1) * up.width * up.height) - 1;
					cv::Mat plane = row.colRange(planeStart, planeEnd);
					h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
				}
			}

			for (int crop_i = 0; crop_i < BATCH; crop_i++)
			{
				for (int i = 0; i < CV_MAT_CN(NPP_OUTPUT); i++)
				{
					cv::Mat cvRes = h_cvResults[crop_i].at(i);
					cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
					diff = cv::abs(cvRes - cvGSRes);
					bool passedThisTime = checkResults<CV_MAT_DEPTH(NPP_OUTPUT)>(diff.cols, diff.rows, diff);
					passed &= passedThisTime;
				}
			}
		}
		catch (const cv::Exception &e)
		{
			if (e.code != -210)
			{
				error_s << e.what();
				passed = false;
				exception = true;
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
				ss << "test_batchresize_x_split3D<" << cvTypeToString<NPP_INPUT>() << ", " << cvTypeToString<NPP_OUTPUT>();
				std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
			}
			else
			{
				std::stringstream ss;
				ss << "test_batchresize_x_split3D<" << cvTypeToString<NPP_INPUT>() << ", " << cvTypeToString<NPP_OUTPUT>();
				std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
			}
		}
	}

	return passed;
}

template <const uint NPP_INPUT, const uint NPP_OUTPUT, const uint NPP_CHANNELS, size_t... Is>
bool test_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cudaStream_t stream, bool enabled)
{
	bool passed = true;
	int dummy[] = {(passed &= test_batchresize_x_split3D<NPP_INPUT, NPP_OUTPUT, NPP_CHANNELS, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, stream, enabled), 0)...};
	return passed;
}

int launch()
{
	constexpr size_t NUM_ELEMS_X = 3840;
	constexpr size_t NUM_ELEMS_Y = 2160;

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	std::unordered_map<std::string, bool> results;
	results["benchmark_npp_batchresize_x_split3D"] = true;
	results["test_batchresize_x_split3D"] = true;
	std::make_index_sequence<batchValues.size()> iSeq{};

#ifdef ENABLE_BENCHMARK
	// Warming up for the benchmarks
	results["benchmark_npp_batchresize_x_split3D"] &= benchmark_npp_batchresize_x_split3D<CV_8UC3, CV_32FC3, 5>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);
#endif

#define LAUNCH_TESTS(NPP_INPUT, NPP_OUTPUT, NPP_CHANNELS) \
	results["test_batchresize_x_split3D"] &= test_batchresize_x_split3D<NPP_INPUT, NPP_OUTPUT, NPP_CHANNELS>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);

	// ouput always   : NPP_32F, num channles idendical: either NPP_CH_3 (RGB) or NPP_CH_A4 (alpha)
	LAUNCH_TESTS(NPP_8U, NPP_32F, NPP_CH_3)
	LAUNCH_TESTS(NPP_8U, NPP_32F, NPP_CH_A4)
	LAUNCH_TESTS(NPP_16U, NPP_32F, NPP_CH_3)
	LAUNCH_TESTS(NPP_16U, NPP_32F, NPP_CH_A4)
	LAUNCH_TESTS(NPP_16S, NPP_32F, NPP_CH_3)
	LAUNCH_TESTS(NPP_16S, NPP_32F, NPP_CH_A4)

#undef LAUNCH_TESTS

	CLOSE_BENCHMARK

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