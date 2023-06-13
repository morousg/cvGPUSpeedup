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
    constexpr int WIDTH = 32;
    constexpr int HEIGHT = 32;
    constexpr int BATCH = 15;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    fk::ReadDeviceFunction<fk::CircularBatchRead<fk::PerThreadRead<fk::_2D, uchar3>, BATCH>> circularBatchRead;
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uchar3>> write3D;

    std::vector<fk::Ptr2D<uchar3>> h_inputAllocations;

    // TODO: fill the host pointers with the number of batch

    std::vector<fk::Ptr2D<uchar3>> inputAllocations;
    std::array<fk::RawPtr<fk::_2D, uchar3>, BATCH> input;
    fk::Tensor<uchar3> output;

    for (int i = 0; i < BATCH; i++) {
        fk::Ptr2D<uchar3> temp(WIDTH, HEIGHT);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }
    output.allocTensor(WIDTH, HEIGHT, BATCH);

    dim3 block = inputAllocations[0].getBlockSize();
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)WIDTH), BATCH };
    fk::cuda_transform << <grid, block, 0, stream >> > (circularBatchRead, write3D);

    gpuErrchk(cudaStreamSynchronize(stream));

    return true;
}

bool testDivergentBatch() {
    constexpr int WIDTH = 32;
    constexpr int HEIGHT = 32;
    constexpr int BATCH = 2;

    cudaStream_t stream;

    gpuErrchk(cudaStreamCreate(&stream));

    std::vector<fk::Ptr2D<uchar3>> h_inputAllocations;

    // TODO: fill the host pointers with the number of batch

    std::vector<fk::Ptr2D<uint>> inputAllocations;
    std::array<fk::RawPtr<fk::_2D, uint>, BATCH> input;
    fk::Tensor<uint> output;

    for (int i = 0; i < BATCH; i++) {
        fk::Ptr2D<uint> temp(WIDTH, HEIGHT);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }
    output.allocTensor(WIDTH, HEIGHT, BATCH);

    auto opSeq1 = fk::buildOperationSequence(fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, uint>> { input[0] },
                                             fk::BinaryDeviceFunction<fk::BinarySum<uint>> {3},
                                             fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uint>> { output.ptr() });
    auto opSeq2 = fk::buildOperationSequence(fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, uint>> { input[1] },
                                             fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, uint>> { output.ptr() });

    dim3 block = inputAllocations[0].getBlockSize();
    dim3 grid{ (uint)ceil((float)WIDTH / (float)block.x), (uint)ceil((float)HEIGHT / (float)WIDTH), BATCH };
    const int opSeqSelection[BATCH] = { 1, 2 };
    fk::cuda_transform_divergent_batch<BATCH><<<grid, block, 0, stream>>>(opSeqSelection, opSeq1, opSeq2);

    gpuErrchk(cudaStreamSynchronize(stream));

    return true;
}

int main() {
    bool worked = testCircularBatchRead();

    return 0;

}