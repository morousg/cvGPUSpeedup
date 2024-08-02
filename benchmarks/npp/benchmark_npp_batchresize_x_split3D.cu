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
#include <opencv2/cudaimgproc.hpp>

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
template <int CV_TYPE_I, int CV_TYPE_O, int BATCH>
bool test_npp_batchresize_x_split3D_OCVBatch(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled) {
		struct Parameters {
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
			{{2u, 37u, 128u, 20u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}
		};

		cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
		cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
		cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_sub;
		cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_div;

		try {
			cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, CV_TYPE_I, val_init);
			std::array<cv::Rect2d, BATCH> crops_2d;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				crops_2d[crop_i] = cv::Rect2d(cv::Point2d(crop_i, crop_i), cv::Point2d(crop_i + 60, crop_i + 120));
			}



			cv::Size up(64, 128);
			cv::cuda::GpuMat d_up(up, CV_TYPE_I);

			std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
			std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

			cv::cuda::GpuMat d_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));
			d_tensor_output.step = up.width * up.height * CV_MAT_CN(CV_TYPE_O) * sizeof(BASE_CUDA_T(CV_TYPE_O));

			cv::cuda::GpuMat d_resize_output(BATCH, up.width * up.height, CV_TYPE_I);
			d_resize_output.step = up.width * up.height * CV_MAT_CN(CV_TYPE_I) * sizeof(BASE_CUDA_T(CV_TYPE_I));

			std::array<cv::cuda::GpuMat, BATCH> d_resized_array;

			cv::cuda::GpuMat d_temp(BATCH, up.width * up.height, CV_TYPE_O);
			d_temp.step = up.width * up.height * CV_MAT_CN(CV_TYPE_O) * sizeof(BASE_CUDA_T(CV_TYPE_O));

			cv::cuda::GpuMat d_temp2(BATCH, up.width * up.height, CV_TYPE_O);
			d_temp2.step = up.width * up.height * CV_MAT_CN(CV_TYPE_O) * sizeof(BASE_CUDA_T(CV_TYPE_O));

			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cv;

			cv::Mat diff(up, CV_MAT_DEPTH(CV_TYPE_O));
			cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));

			std::array<cv::cuda::GpuMat, BATCH> crops;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				crops[crop_i] = d_input(crops_2d[crop_i]);
				d_resized_array[crop_i] = cv::cuda::GpuMat(up, d_resize_output.type(), d_resize_output.row(crop_i).data);
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_I); i++) {
					d_output_cv.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
					h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
				}
			}

			constexpr bool correctDept = CV_MAT_DEPTH(CV_TYPE_O) == CV_32F;
	 
			START_OCV_BENCHMARK

				//NPP

				 // cvGPUSpeedup version+
		   //     cudaStream_t stream;

			 //   gpuErrchk(cudaStreamCreate(&stream));

				NppStreamContext nppcontext = initNppStreamContext(static_cast<cudaStream_t>(cv_stream.cudaPtr()));
			if constexpr (CV_MAT_CN(CV_TYPE_O) == 3 && correctDept) {
	

				//                NPP_CHECK(nppiResizeBatch_8u_C3R_Ctx(NppiSize(1,1), NppiSize(up)))
								//nppiResizeBatch_8u_C3R_Ctx()
								//nppiSwapChannels_8u_C3IR_Ctx()             
								//nppiConvert_8u32f_AC3()
								//nppiMulC_8u_C3IRSfs_Ctx()
								//nppiSubC_8u_C3RSfs_Ctx/()
								//nppiDivC_8u_C3RSfs_Ctx()
								//split????

			}
			else if constexpr (CV_MAT_CN(CV_TYPE_O) == 4 && correctDept) {
				//nppiSwapChannels_32f_AC4R
			   //nppiResizeBatch_32f_AC4R_Ctx()
				//nppiConvert_8u32f_AC4R()
			}
			else {

			}

			//     gpuErrchk(cudaStreamDestroy(stream));
			STOP_OCV_START_CVGS_BENCHMARK
				// cvGPUSpeedup version
				if constexpr (CV_MAT_CN(CV_TYPE_O) == 3 && correctDept) {
					const auto resizeDF = cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH);
					cvGS::executeOperations(cv_stream,
						resizeDF,
						cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_TYPE_O>(),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
				else if constexpr (CV_MAT_CN(CV_TYPE_O) == 4 && correctDept) {
					cvGS::executeOperations(cv_stream,
						cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
						cvGS::cvtColor<cv::COLOR_RGBA2BGRA, CV_TYPE_O>(),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
				else {
					cvGS::executeOperations(cv_stream,
						cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
						cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
			STOP_CVGS_BENCHMARK
				d_tensor_output.download(h_tensor_output, cv_stream);

			// Verify results
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					d_output_cv[crop_i].at(i).download(h_cvResults[crop_i].at(i), cv_stream);
				}
			}

			cv_stream.waitForCompletion();

			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				cv::Mat row = h_tensor_output.row(crop_i);
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					int planeStart = i * up.width * up.height;
					int planeEnd = ((i + 1) * up.width * up.height) - 1;
					cv::Mat plane = row.colRange(planeStart, planeEnd);
					h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
				}
			}

			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					cv::Mat cvRes = h_cvResults[crop_i].at(i);
					cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
					diff = cv::abs(cvRes - cvGSRes);
					bool passedThisTime = checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
					passed &= passedThisTime;
				}
			}

		}
		catch (const cv::Exception& e) {
			if (e.code != -210) {
				error_s << e.what();
				passed = false;
				exception = true;
			}
		}
		catch (const std::exception& e) {
			error_s << e.what();
			passed = false;
			exception = true;
		}

		if (!passed) {
			if (!exception) {
				std::stringstream ss;
				ss << "test_npp_batchresize_x_split3D_OCVBatch<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
			}
			else {
				std::stringstream ss;
				ss << "test_npp_batchresize_x_split3D_OCVBatch<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
			}
		}
	}

	return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool test_npp_batchresize_x_split3D_OCVBatch(int NUM_ELEMS_X, int NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream& cv_stream, bool enabled) {
	bool passed = true;
	int dummy[] = { (passed &= test_npp_batchresize_x_split3D_OCVBatch<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
	return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, int BATCH>
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled) {
		struct Parameters {
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
			{{2u, 37u, 128u, 20u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}
		};

		cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
		cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
		cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_sub;
		cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_div;

		try {
			cv::cuda::GpuMat d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
			std::array<cv::Rect2d, BATCH> crops_2d;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				crops_2d[crop_i] = cv::Rect2d(cv::Point2d(crop_i, crop_i), cv::Point2d(crop_i + 60, crop_i + 120));
			}

			cv::Size up(64, 128);
			cv::cuda::GpuMat d_up(up, CV_TYPE_I);
			cv::cuda::GpuMat d_temp(up, CV_TYPE_O);
			cv::cuda::GpuMat d_temp2(up, CV_TYPE_O);

			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cv;
			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cvGS;
			std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
			std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

			cv::cuda::GpuMat d_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));
			d_tensor_output.step = up.width * up.height * CV_MAT_CN(CV_TYPE_O) * sizeof(BASE_CUDA_T(CV_TYPE_O));

			cv::Mat diff(up, CV_MAT_DEPTH(CV_TYPE_O));
			cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));

			std::array<cv::cuda::GpuMat, BATCH> crops;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				crops[crop_i] = d_input(crops_2d[crop_i]);
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_I); i++) {
					d_output_cv.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
					h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
				}
			}

			constexpr bool correctDept = CV_MAT_DEPTH(CV_TYPE_O) == CV_32F;

			START_OCV_BENCHMARK
				// NPP version

				//cudaStream_t stream;
			//gpuErrchk(cudaStreamCreate(&stream));
			
			 			NppStreamContext nppcontext = initNppStreamContext(static_cast<cudaStream_t>(cv_stream.cudaPtr()));
			//NppiResizeBatchCXR batch
			if constexpr (CV_MAT_CN(CV_TYPE_I) == 3 && correctDept) {
				//                nppiResizeBatch_8u_C3R_Ctx()
						 //nppiSwapChannels_8u_C3IR_Ctx()             
						 //nppiConvert_8u32f_AC3()
						 //nppiMulC_8u_C3IRSfs_Ctx()
						 //nppiSubC_8u_C3RSfs_Ctx/()
						 //nppiDivC_8u_C3RSfs_Ctx()
						 //split????

			}
			else if constexpr (CV_MAT_CN(CV_TYPE_I) == 4 && correctDept) {

			}
			else {

			}
//			gpuErrchk(cudaStreamDestroy(stream));
			STOP_OCV_START_CVGS_BENCHMARK
				// cvGPUSpeedup
				if constexpr (CV_MAT_CN(CV_TYPE_I) == 3 && correctDept) {
					cvGS::executeOperations(cv_stream,
						cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
						cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_TYPE_O>(),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
				else if constexpr (CV_MAT_CN(CV_TYPE_I) == 4 && correctDept) {
					cvGS::executeOperations(cv_stream,
						cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
						cvGS::cvtColor<cv::COLOR_RGBA2BGRA, CV_TYPE_O>(),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
				else {
					cvGS::executeOperations(cv_stream,
						cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
						cvGS::multiply<CV_TYPE_O>(val_alpha),
						cvGS::subtract<CV_TYPE_O>(val_sub),
						cvGS::divide<CV_TYPE_O>(val_div),
						cvGS::split<CV_TYPE_O>(d_tensor_output, up));
				}
			STOP_CVGS_BENCHMARK
				d_tensor_output.download(h_tensor_output, cv_stream);

			// Verify results
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					d_output_cv[crop_i].at(i).download(h_cvResults[crop_i].at(i), cv_stream);
				}
			}

			cv_stream.waitForCompletion();

			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				cv::Mat row = h_tensor_output.row(crop_i);
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					int planeStart = i * up.width * up.height;
					int planeEnd = ((i + 1) * up.width * up.height) - 1;
					cv::Mat plane = row.colRange(planeStart, planeEnd);
					h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
				}
			}

			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
					cv::Mat cvRes = h_cvResults[crop_i].at(i);
					cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
					diff = cv::abs(cvRes - cvGSRes);
					bool passedThisTime = checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
					passed &= passedThisTime;
				}
			}
		}
		catch (const cv::Exception& e) {
			if (e.code != -210) {
				error_s << e.what();
				passed = false;
				exception = true;
			}
		}
		catch (const std::exception& e) {
			error_s << e.what();
			passed = false;
			exception = true;
		}

		if (!passed) {
			if (!exception) {
				std::stringstream ss;
				ss << "test_npp_batchresize_x_split3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
			}
			else {
				std::stringstream ss;
				ss << "test_npp_batchresize_x_split3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
				std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
			}
		}
	}

	return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
	bool passed = true;
	int dummy[] = { (passed &= test_npp_batchresize_x_split3D<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
	return passed;
}

int launch() {
	constexpr size_t NUM_ELEMS_X = 3840;
	constexpr size_t NUM_ELEMS_Y = 2160;

	cv::cuda::Stream cv_stream;

	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	std::unordered_map<std::string, bool> results;
	results["test_npp_batchresize_x_split3D_OCVBatch"] = true;
	results["test_npp_batchresize_x_split3D"] = true;
	std::make_index_sequence<batchValues.size()> iSeq{};

#ifdef ENABLE_BENCHMARK
	// Warming up for the benchmarks
	results["test_npp_batchresize_x_split3D_OCVBatch"] &= test_npp_batchresize_x_split3D_OCVBatch<CV_8UC3, CV_32FC3, 5>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);
#endif

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_npp_batchresize_x_split3D_OCVBatch"] &= test_npp_batchresize_x_split3D_OCVBatch<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream, true); \
    results["test_npp_batchresize_x_split3D"] &= test_npp_batchresize_x_split3D<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream, true);

	LAUNCH_TESTS(CV_8UC3, CV_32FC3)
		LAUNCH_TESTS(CV_8UC4, CV_32FC4)
		LAUNCH_TESTS(CV_16UC3, CV_32FC3)
		LAUNCH_TESTS(CV_16UC4, CV_32FC4)
		LAUNCH_TESTS(CV_16SC3, CV_32FC3)
		LAUNCH_TESTS(CV_16SC4, CV_32FC4)

#undef LAUNCH_TESTS

		CLOSE_BENCHMARK

		int returnValue = 0;
	for (const auto& [key, passed] : results) {
		if (passed) {
			std::cout << key << " passed!!" << std::endl;
		}
		else {
			std::cout << key << " failed!!" << std::endl;
			returnValue = -1;
		}
	}

	return returnValue;
}