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

#include <iostream>

#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/algorithms/image_processing/saturate.cuh>
#include "tests/main.h"

template <typename T>
bool testPtr_2D() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    constexpr size_t width_crop = 300;
    constexpr size_t height_crop = 200;

    fk::Point startPoint = {100, 200};

    fk::Ptr2D<T> input(width, height);
    fk::Ptr2D<T> cropedInput = input.crop(startPoint, fk::PtrDims<fk::_2D>(width_crop, height_crop));
    fk::Ptr2D<T> output(width_crop, height_crop);
    fk::Ptr2D<T> outputBig(width, height);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    dim3 block2D(32,8);
    dim3 grid2D((uint)std::ceil(width_crop / (float)block2D.x),
                (uint)std::ceil(height_crop / (float)block2D.y));
    dim3 grid2DBig((uint)std::ceil(width / (float)block2D.x),
                   (uint)std::ceil(height / (float)block2D.y));

    dim3 gridActiveThreadsCrop(cropedInput.dims().width, cropedInput.dims().height);
    dim3 gridActiveThreads(input.dims().width, input.dims().height);
    fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, T>> readCrop{cropedInput, gridActiveThreadsCrop};
    fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, T>> readFull{input, gridActiveThreads};

    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, T>> opFinal_2D = { output };
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, T>> opFinal_2DBig = { outputBig };

    for (int i=0; i<100; i++) {
        fk::cuda_transform<<<grid2D, block2D, 0, stream>>>(readCrop, opFinal_2D);
        fk::cuda_transform<<<grid2DBig, block2D, 0, stream>>>(readFull, opFinal_2DBig);
    }

    cudaError_t err = cudaStreamSynchronize(stream);

    // TODO: use some values and check results correctness

    if (err != cudaSuccess) {
        return false;
    } else {
        return true;
    }
}

int launch() {
    bool test2Dpassed = true;

    test2Dpassed &= testPtr_2D<uchar>();
    test2Dpassed &= testPtr_2D<uchar3>();
    test2Dpassed &= testPtr_2D<float>();
    test2Dpassed &= testPtr_2D<float3>();

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    fk::Ptr2D<uchar> input(64,64);
    fk::Ptr2D<uint> output(64,64);
    
    dim3 gridActiveThreads(64, 64);
    fk::Read<fk::PerThreadRead<fk::_2D, uchar>> read{ input, gridActiveThreads };
    fk::Unary<fk::SaturateCast<uchar, uint>> cast = {};
    fk::Write<fk::PerThreadWrite<fk::_2D, uint>> write { output };

    auto fusedDF = fk::fuseDF(read, cast, fk::Binary<fk::Mul<uint>>{4});
    fusedDF.params.params;
    //fusedDF.params.next.params; // Should not compile
    fusedDF.params.next.next.params;

    fk::cuda_transform<<<dim3(1,8),dim3(64,8),0,stream>>>(fusedDF, write);

    fk::OperationTuple<fk::PerThreadRead<fk::_2D, uchar>, fk::SaturateCast<uchar, uint>, fk::PerThreadWrite<fk::_2D, uint>> myTup{};

    fk::get_params<2>(myTup);
    constexpr bool test1 = std::is_same_v<fk::get_type_t<0, decltype(myTup)>, fk::PerThreadRead<fk::_2D, uchar>>;
    constexpr bool test2 = std::is_same_v<fk::get_type_t<1, decltype(myTup)>, fk::SaturateCast<uchar, uint>>;
    constexpr bool test3 = std::is_same_v<fk::get_type_t<2, decltype(myTup)>, fk::PerThreadWrite<fk::_2D, uint>>;

    gpuErrchk(cudaStreamSynchronize(stream));

    if (test2Dpassed && fk::and_v<test1, test2, test3>) {
        std::cout << "cuda_transform executed!!" << std::endl; 
    } else {
        std::cout << "cuda_transform failed!!" << std::endl;
    }

    return 0;
}