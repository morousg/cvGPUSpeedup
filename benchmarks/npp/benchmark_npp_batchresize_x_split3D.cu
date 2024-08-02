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
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cudaStream_t& cv_stream, bool enabled) {
	std::stringstream error_s;
	bool passed = true;
	bool exception = false;

	if (enabled) {
		float alpha = 0.3f;

		uchar3 val_init = { 5u, 5u, 5u };
		float3 val_alpha = { alpha, alpha, alpha };
		float3 val_sub = { 1.f, 4.f, 3.2f };
		float3 val_div = { 1.f, 4.f, 3.2f };

		try {
			fk::Ptr2D<uchar3> d_input((int)NUM_ELEMS_X, (int)NUM_ELEMS_Y);

			

			std::array<cv::Rect2d, BATCH> crops_2d;
			std::array<NppiImageDescriptor, BATCH> srccropsdesc;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
			    const Ptr2D<uchar3> d_cropped = d_input.crop2D(fk::Point(crop_i, crop_i), fk::PtrDims<fk::_2D>{60, 120, d_input.ptr().dims.pitch});
				srccropsdesc[crop_i].pData = (uchar*)d_input(crops_2d[crop_i]).data;
				srccropsdesc[crop_i].nStep = d_input(crops_2d[crop_i]).step;
				srccropsdesc[crop_i].oSize = { crop_i + 60, crop_i + 120 };


			}

			cv::Size up(64, 128);
			cv::cuda::GpuMat d_up(up, CV_TYPE_I);
			cv::cuda::GpuMat d_temp(up, CV_TYPE_O);
			cv::cuda::GpuMat d_temp2(up, CV_TYPE_O);

			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cv;
			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_npp;
			std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cvGS;
			std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
			std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

			cv::cuda::GpuMat d_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));
			d_tensor_output.step = up.width * up.height * CV_MAT_CN(CV_TYPE_O) * sizeof(BASE_CUDA_T(CV_TYPE_O));

			cv::Mat diff(up, CV_MAT_DEPTH(CV_TYPE_O));
			cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));

			std::array<cv::cuda::GpuMat, BATCH> crops;
			
			std::array<NppiImageDescriptor, BATCH> dstcropsdesc;
			for (int crop_i = 0; crop_i < BATCH; crop_i++) {
				crops[crop_i] = d_input(crops_2d[crop_i]);
				dstcropsdesc[crop_i].pData = (uchar*)d_output_npp[crop_i].data();
				
				dstcropsdesc[crop_i].nStep = d_output_npp[crop_i].step;
				dstcropsdesc[crop_i].oSize = { up.width,up.height };

				for (int i = 0; i < CV_MAT_CN(CV_TYPE_I); i++) {

					d_output_cv.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
					h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
				}
			}

			constexpr bool correctDept = CV_MAT_DEPTH(CV_TYPE_O) == CV_32F;

			
				// NPP version

				//cudaStream_t stream;
			//gpuErrchk(cudaStreamCreate(&stream));

				NppStreamContext nppcontext = initNppStreamContext(static_cast<cudaStream_t>(cv_stream.cudaPtr()));

			if constexpr (CV_MAT_CN(CV_TYPE_I) == 3 && correctDept) {
				//(for i=0;i<BATCH;++i)
				//nppiConvert_8u32f_AC3
				// 
				//nppiResizeBatch_32f_C3R_Ctx()
				
				// //(for i=0;i<BATCH;++i)
					//nppiSwapChannels_32f_C3IR_Ctx()             
				 	 //nppiMulC_32f_C3IRSfs_Ctx()
						 //nppiSubC_32f_C3RSfs_Ctx/()
						 //nppiDivC_32f_C3RSfs_Ctx()
						 //split???? nppiCopy_32f_C3P3R_Ctx

			}
			else if constexpr (CV_MAT_CN(CV_TYPE_I) == 4 && correctDept) {

			}
			else {

			}
			//			gpuErrchk(cudaStreamDestroy(stream));
			
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

template <size_t... Is>
bool test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cudaStream_t cv_stream, bool enabled) {
	bool passed = true;
	int dummy[] = { (passed &= test_npp_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
	return passed;
}

int launch() {
	constexpr size_t NUM_ELEMS_X = 3840;
	constexpr size_t NUM_ELEMS_Y = 2160;

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	std::unordered_map<std::string, bool> results;
	results["test_npp_batchresize_x_split3D"] = true;
	std::make_index_sequence<batchValues.size()> iSeq{};

    results["test_npp_batchresize_x_split3D"] &= test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);

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