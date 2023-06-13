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
#include <cvGPUSpeedup.h>
#include <opencv2/cudaimgproc.hpp>

bool testCircularBatchRead() {
    constexpr int WIDTH = 1;
    constexpr int HEIGHT = 1;
    constexpr int BATCH = 15;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    std::vector<fk::Ptr2D<uchar3>> h_inputAllocations;

    for (uint i = 0; i < BATCH; i++) {
        fk::Ptr2D<uchar3> temp(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const fk::Point p{ x, y, 0 };
                *fk::PtrAccessor<fk::_2D>::point(p, temp.ptr()) = fk::make_<uchar3>(i, i, i);
            }
        }
        h_inputAllocations.push_back(temp);
    }

    std::vector<fk::Ptr2D<uchar3>> inputAllocations;
    std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> input;
    fk::Tensor<uchar3> output;

    for (int i = 0; i < BATCH; i++) {
        fk::Ptr2D<uchar3> temp(WIDTH, HEIGHT);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }
    output.allocTensor(WIDTH, HEIGHT, BATCH);

    fk::ReadDeviceFunction<fk::CircularBatchRead<fk::PerThreadRead<fk::_2D, uchar3>, BATCH>> circularBatchRead;
    circularBatchRead.activeThreads = {WIDTH, HEIGHT, BATCH};
    circularBatchRead.params.first = 4;
    for (int i = 0; i < BATCH; i++) {
        circularBatchRead.params.params[i] = input[i];
    }
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uchar3>> write3D{output};

    dim3 block = inputAllocations[0].getBlockSize();
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)block.y), BATCH };
    fk::cuda_transform << <grid, block, 0, stream >> > (circularBatchRead, write3D);

    gpuErrchk(cudaStreamSynchronize(stream));

    return true;
}

bool testDivergentBatch() {
    constexpr int WIDTH = 1;
    constexpr int HEIGHT = 1;
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
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)WIDTH), BATCH };
    const int opSeqSelection[BATCH] = { 1, 2 };
    fk::cuda_transform_divergent_batch<BATCH><<<grid, block, 0, stream>>>(opSeqSelection, opSeq1, opSeq2);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpyAsync(h_output.ptr().data, output.ptr().data, output.sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    for (uint z = 0; z < BATCH; z++) {
        if (z == 0) {
            for (uint y = 0; y < HEIGHT; y++) {
                for (uint x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x, y, z};
                    correct &= *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr()) == VAL_SUM;
                }
            }
        }
        else {
            for (uint y = 0; y < HEIGHT; y++) {
                for (uint x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x, y, z};
                    correct &= *fk::PtrAccessor<fk::_3D>::point(p, h_groundTruth.ptr()) == z;
                }
            }
        }
    }

    return correct;
}

int main() {
    /*if (testCircularBatchRead()) {
        std::cout << "testCircularBatchRead OK" << std::endl;
    } else {
        std::cout << "testCircularBatchRead Failed!" << std::endl;
    }*/
    if (testDivergentBatch()) {
        std::cout << "testDivergentBatch OK" << std::endl;
    } else {
        std::cout << "testDivergentBatch Failed!" << std::endl;
    }

    return 0;
}