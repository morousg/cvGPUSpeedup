/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "testsCommon.cuh"
#include <cvGPUSpeedup.cuh>

bool testCircularBatchRead() {
    constexpr uint WIDTH = 32;
    constexpr uint HEIGHT = 32;
    constexpr uint BATCH = 15;
    constexpr uint FIRST = 4;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    std::vector<fk::Ptr2D<uchar3>> h_inputAllocations;

    std::vector<fk::Ptr2D<uchar3>> inputAllocations;
    std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> input;
    fk::Tensor<uchar3> output;
    fk::Tensor<uchar3> h_output;

    for (int i = 0; i < BATCH; i++) {
        fk::Ptr2D<uchar3> h_temp(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const fk::Point p{ x, y, 0 };
                *fk::PtrAccessor<fk::_2D>::point(p, h_temp.ptr()) = fk::make_<uchar3>(i, i, i);
            }
        }
        h_inputAllocations.push_back(h_temp);
        fk::Ptr2D<uchar3> temp(WIDTH, HEIGHT);
        inputAllocations.push_back(temp);
        input[i] = temp;
        gpuErrchk(cudaMemcpy2DAsync(temp.ptr().data, temp.dims().pitch, h_temp.ptr().data, h_temp.dims().pitch,
            h_temp.dims().width * sizeof(uchar3), h_temp.dims().height, cudaMemcpyHostToDevice, stream));
    }
    output.allocTensor(WIDTH, HEIGHT, BATCH);
    h_output.allocTensor(WIDTH, HEIGHT, BATCH, 1, fk::MemType::HostPinned);

    fk::ReadDeviceFunction<fk::CircularBatchRead<fk::CircularDirection::Ascendent, fk::PerThreadRead<fk::_2D, uchar3>, BATCH>> circularBatchRead;
    circularBatchRead.activeThreads = {WIDTH, HEIGHT, BATCH};
    circularBatchRead.params.first = FIRST;
    for (int i = 0; i < BATCH; i++) {
        circularBatchRead.params.params[i] = input[i];
    }
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uchar3>> write3D{output};

    dim3 block = inputAllocations[0].getBlockSize();
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)block.y), BATCH };
    fk::cuda_transform << <grid, block, 0, stream >> > (circularBatchRead, write3D);

    gpuErrchk(cudaMemcpyAsync(h_output.ptr().data, output.ptr().data, output.sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    for (uint z = 0; z < BATCH; z++) {
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                fk::Point p{ x, y, z };
                uchar3 res = *fk::PtrAccessor<fk::_3D>::point(p, h_output.ptr());
                uchar newZ = (z + FIRST);
                uchar3 gt  = newZ >= BATCH ? fk::make_set<uchar3>(newZ - BATCH) : fk::make_set<uchar3>(newZ);
                correct &= res.x == gt.x;
                correct &= res.y == gt.y;
                correct &= res.z == gt.z;
            }
        }
    }

    return correct;
}

bool testDivergentBatch() {
    constexpr uint WIDTH = 32;
    constexpr uint HEIGHT = 32;
    constexpr uint BATCH = 2;
    constexpr uint VAL_SUM = 3;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    std::vector<fk::Ptr2D<uint>> h_inputAllocations;
    std::vector<fk::Ptr2D<uint>> inputAllocations;
    std::array<fk::RawPtr<fk::_2D, uint>, BATCH> input;
    fk::Tensor<uint> output;
    fk::Tensor<uint> h_output;
    fk::Tensor<uint> h_groundTruth;

    for (uint i = 0; i < BATCH; i++) {
        fk::Ptr2D<uint> h_temp(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);
        h_temp.setTo(i);
        h_inputAllocations.push_back(h_temp);
        fk::Ptr2D<uint> temp(WIDTH, HEIGHT);
        inputAllocations.push_back(temp);
        input[i] = temp;
        gpuErrchk(cudaMemcpy2DAsync(temp.ptr().data, temp.ptr().dims.pitch,
                                    h_temp.ptr().data, h_temp.ptr().dims.pitch,
                                    h_temp.dims().width * sizeof(uint), h_temp.dims().height,
                                    cudaMemcpyHostToDevice, stream));
    }

    output.allocTensor(WIDTH, HEIGHT, BATCH);
    h_output.allocTensor(WIDTH, HEIGHT, BATCH, 1, fk::MemType::HostPinned);
    h_groundTruth.allocTensor(WIDTH, HEIGHT, BATCH, 1, fk::MemType::HostPinned);

    for (uint z = 0; z < BATCH; z++) {
        if (z == 0) {
            for (uint y = 0; y < HEIGHT; y++) {
                for (uint x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x,y,z};
                    *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr()) = VAL_SUM;
                }
            }
        } else {
            for (uint y = 0; y < HEIGHT; y++) {
                for (uint x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x, y, z};
                    *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr()) = z;
                }
            }
        }
    }

    auto opSeq1 = fk::buildOperationSequence(fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, uint>> { input[0], { WIDTH, HEIGHT, BATCH } },
                                             fk::BinaryDeviceFunction<fk::BinarySum<uint>> {VAL_SUM},
                                             fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uint>> { output.ptr() });
    auto opSeq2 = fk::buildOperationSequence(fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, uint>> { input[1], { WIDTH, HEIGHT, BATCH } },
                                             fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uint>> { output.ptr() });

    dim3 block = inputAllocations[0].getBlockSize();
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)block.y), BATCH };
    const fk::Array<int, BATCH> opSeqSelection{ {1, 2} };
    fk::cuda_transform_divergent_batch<BATCH><<<grid, block, 0, stream>>>(opSeqSelection, opSeq1, opSeq2);

    gpuErrchk(cudaMemcpyAsync(h_output.ptr().data, output.ptr().data, output.sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    for (uint z = 0; z < BATCH; z++) {
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const uint gt = *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr());
                const uint res = *fk::PtrAccessor<fk::_3D>::point(p, h_output.ptr());
                correct &= gt == res;
            }
        }
    }

    return correct;
}

template <typename IT, typename OT>
bool testCircularTensor() {
    using TensorOT = typename fk::VectorTraits<OT>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = fk::cn<IT>;
    constexpr int ITERS = 100;

    fk::CircularTensor<TensorOT, COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst, fk::ColorPlanes::Standard> myTensor(WIDTH, HEIGHT);
    fk::Tensor<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::Ptr2D<IT> input(WIDTH, HEIGHT);
    fk::Ptr2D<IT> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    h_myTensor.setTo(10.0f);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        h_input.setTo(fk::make_<IT>(i + 1, i + 1, i + 1));
        gpuErrchk(cudaMemcpy2DAsync(input.ptr().data, input.ptr().dims.pitch,
            h_input.ptr().data, h_input.ptr().dims.pitch,
            h_input.ptr().dims.width * sizeof(IT),
            h_input.ptr().dims.height,
            cudaMemcpyHostToDevice, stream));
        myTensor.update(stream, fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, IT>> {input.ptr(), { WIDTH, HEIGHT, 1 }},
            fk::UnaryDeviceFunction<fk::UnaryCast<IT, OT>> {},
            fk::WriteDeviceFunction<fk::TensorSplitWrite<OT>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    for (uint z = 0; z < BATCH; z++) {
        const TensorOT value = (TensorOT)(ITERS - z);
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const TensorOT res = *fk::PtrAccessor<fk::_3D>::point(p, h_myTensor.ptr());
                correct &= value == res;
            }
        }
    }

    return correct;
}

template <int IT, int OT>
bool testCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH =128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);
    constexpr int ITERS = 100;

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst> myTensor(WIDTH, HEIGHT);
    fk::Tensor<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    h_myTensor.setTo(10.0f);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        h_input.setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1));
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
                                    h_input.ptr().data, h_input.ptr().dims.pitch,
                                    h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
                                    h_input.ptr().dims.height,
                                    cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
                        fk::UnaryDeviceFunction<fk::UnaryCast<CUDA_T(IT), CUDA_T(OT)>> {},
                        fk::WriteDeviceFunction<fk::TensorSplitWrite<CUDA_T(OT)>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const size_t plane_pixels = h_myTensor.dims().width * h_myTensor.dims().height;
    for (uint z = 0; z < BATCH; z++) {
        const TensorOT value = (TensorOT)(ITERS - z);
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const TensorOT* workPlane = fk::PtrAccessor<fk::_3D>::point(p, h_myTensor.ptr());
                const TensorOT resX = *workPlane;
                correct &= value == resX;
                const TensorOT resY = *(workPlane + plane_pixels);
                correct &= value == resY;
                const TensorOT resZ = *(workPlane + (plane_pixels * 2));
                correct &= value == resZ;
            }
        }
    }

    return correct;
}

template <int IT, int OT>
bool testTransposedCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);
    constexpr int ITERS = 100;

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst, fk::ColorPlanes::Transposed> myTensor(WIDTH, HEIGHT);
    fk::TensorT<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::TensorT<TensorOT> h_myInternalTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    h_myTensor.setTo(10.0f);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        h_input.setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1));
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
            h_input.ptr().data, h_input.ptr().dims.pitch,
            h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
            h_input.ptr().dims.height,
            cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
            fk::UnaryDeviceFunction<fk::UnaryCast<CUDA_T(IT), CUDA_T(OT)>> {},
            fk::WriteDeviceFunction<fk::TensorTSplitWrite<CUDA_T(OT)>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const auto dims = h_myTensor.dims();
    const size_t plane_pixels = dims.width * dims.height;
    for (int cp = 0; cp < (int)dims.color_planes; cp++) {
        for (int y = 0; y < (int)dims.height; y++) {
            for (int z = 0; z < (int)BATCH; z++) {
                const auto* plane = fk::PtrAccessor<fk::T3D>::cr_point(fk::Point(0, 0, z), h_myTensor.ptr())
                    + (plane_pixels * dims.planes * cp);
                for (int x = 0; x < (int)dims.width; x++) {
                    correct &= ITERS - z == plane[x + (y * dims.width)];
                }
            }
        }
    }

    return correct;
}

template <int IT, int OT>
bool testTransposedOldestFirstCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);
    constexpr int ITERS = 100;

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::OldestFirst, fk::ColorPlanes::Transposed> myTensor(WIDTH, HEIGHT);
    fk::TensorT<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::TensorT<TensorOT> h_myInternalTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    h_myTensor.setTo(10.0f);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        h_input.setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1));
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
            h_input.ptr().data, h_input.ptr().dims.pitch,
            h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
            h_input.ptr().dims.height,
            cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
            fk::UnaryDeviceFunction<fk::UnaryCast<CUDA_T(IT), CUDA_T(OT)>> {},
            fk::WriteDeviceFunction<fk::TensorTSplitWrite<CUDA_T(OT)>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const auto dims = h_myTensor.dims();
    const size_t plane_pixels = dims.width * dims.height;
    for (int cp = 0; cp < (int)dims.color_planes; cp++) {
        for (int y = 0; y < (int)dims.height; y++) {
            for (int z = 0; z < (int)BATCH; z++) {
                const auto* plane = fk::PtrAccessor<fk::T3D>::cr_point(fk::Point(0, 0, z), h_myTensor.ptr())
                    + (plane_pixels * dims.planes * cp);
                for (int x = 0; x < (int)dims.width; x++) {
                    correct &= ITERS - (BATCH - z - 1) == plane[x + (y * dims.width)];
                }
            }
        }
    }

    return correct;
}

int main() {
    if (testCircularBatchRead()) {
        std::cout << "testCircularBatchRead OK" << std::endl;
    } else {
        std::cout << "testCircularBatchRead Failed!" << std::endl;
    }
    if (testDivergentBatch()) {
        std::cout << "testDivergentBatch OK" << std::endl;
    } else {
        std::cout << "testDivergentBatch Failed!" << std::endl;
    }
    if (testCircularTensor<uchar3, float3>()) {
        std::cout << "testCircularTensor<uchar3, float3> OK" << std::endl;
    } else {
        std::cout << "testCircularTensor<uchar3, float3> Failed!" << std::endl;
    }
    if (testCircularTensorcvGS<CV_8UC3, CV_32FC3>()) {
        std::cout << "testCircularTensorcvGS<CV_8UC3, CV_32FC3> OK" << std::endl;
    } else {
        std::cout << "testCircularTensorcvGS<CV_8UC3, CV_32FC3> Failed!" << std::endl;
    }
    if (testTransposedCircularTensorcvGS<CV_8UC3, CV_32FC3>()) {
        std::cout << "testTransposedCircularTensorcvGS<CV_8UC3, CV_32FC3> OK" << std::endl;
    } else {
        std::cout << "testTransposedCircularTensorcvGS <CV_8UC3, CV_32FC3> Failed!" << std::endl;
    }
    if (testTransposedOldestFirstCircularTensorcvGS<CV_8UC3, CV_32FC3>()) {
        std::cout << "testTransposedOldestFirstCircularTensorcvGS<CV_8UC3, CV_32FC3> OK" << std::endl;
    } else {
        std::cout << "testTransposedOldestFirstCircularTensorcvGS <CV_8UC3, CV_32FC3> Failed!" << std::endl;
    }

    return 0;
}