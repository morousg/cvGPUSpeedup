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

#include <sstream>

#include "testsCommon.h"
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>

bool testCircularBatchRead() {
    constexpr int WIDTH = 32;
    constexpr int HEIGHT = 32;
    constexpr int BATCH = 15;
    constexpr int FIRST = 4;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    std::vector<fk::Ptr2D<uchar3>> h_inputAllocations;

    for (uint i = 0; i < BATCH; i++) {
        
    }

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

    fk::ReadDeviceFunction<fk::CircularBatchRead<fk::PerThreadRead<fk::_2D, uchar3>, BATCH>> circularBatchRead;
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
                uchar3 gt  = z >= FIRST ? fk::make_set<uchar3>(z - FIRST) : fk::make_set<uchar3>(z + (BATCH - FIRST));
                correct &= res.x == gt.x;
                correct &= res.y == gt.y;
                correct &= res.z == gt.z;
            }
        }
    }

    return correct;
}

bool testDivergentBatch() {
    constexpr int WIDTH = 32;
    constexpr int HEIGHT = 32;
    constexpr int BATCH = 2;
    constexpr int VAL_SUM = 3;

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
    const fk::OperationSequenceSelector<BATCH> opSeqSelection{ {1, 2} };
    fk::cuda_transform_divergent_batch<BATCH><<<grid, block, 0, stream>>>(opSeqSelection, opSeq1, opSeq2);

    gpuErrchk(cudaMemcpyAsync(h_output.ptr().data, output.ptr().data, output.sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    for (uint z = 0; z < BATCH; z++) {
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < HEIGHT; x++) {
                const fk::Point p{x, y, z};
                const uint gt = *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr());
                const uint res = *fk::PtrAccessor<fk::_3D>::point(p, h_output.ptr());
                correct &= gt == res;
            }
        }
    }

    return correct;
}

struct A {};
struct B {};
struct C {};
struct D {};

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

    auto p1 = thrust::make_tuple(1, 2, 3);
    auto [head, tail] = thrust::make_tuple(1, 2, 3, 4, 5, 6);

    int size = thrust::tuple_size<decltype(tail)>::value;

    auto p2 = fk::tuple_cat(p1, thrust::make_tuple(4));

    auto p3 = fk::insert_before_last_tup(5, p2);

    auto p4 = fk::insert_before_last(5, 1, 2, 3, 4, 6);

    auto p5 = fk::buildOperationSequence_tup(p4);

    auto p6 = ([](const auto& elem, const auto&... args)
        { return fk::insert_before_last(elem, args...); })(C{}, A{}, B{}, D{});

    std::cout << fk::TypeIndex<uchar1, fk::VOne>::value << " " <<
        fk::TypeIndex_v<long2, fk::VTwo> << " " <<
        fk::TypeIndex_v<ushort1, fk::VOne> << std::endl;

    std::cout << typeid(fk::TypeFromIndex<0, fk::VOne>::type).name() << std::endl;
    std::cout << typeid(fk::TypeFromIndex_t<2, fk::VOne>).name() << std::endl;

    return 0;
}