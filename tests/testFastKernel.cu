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

#include <fast_kernel/fast_kernel.cuh>

void test_mult_sum_div_float(float* data, dim3 data_dims, cudaStream_t stream) {
    // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid(data_dims.x/512);

    fk::binary_operation_scalar<fk::binary_mul<float>, float> op1 = {5.f};
    fk::binary_operation_pointer<fk::binary_sum<float>, float> op2 = {data};
    fk::binary_operation_scalar<fk::binary_div<float>, float> op3 = {2.f};
    fk::binary_operation_scalar<fk::binary_mul<float>, float> op4 = {5.f};
    fk::binary_operation_scalar<fk::binary_div<float>, float> op5 = {3.f};
    fk::binary_operation_scalar<fk::binary_mul<float>, float> op6 = {7.f};

    fk::cuda_transform<<<grid, thread_block, 0, stream>>>(data_dims.x, data, data, op1, op2, op3, op4, op5, op6);
    gpuErrchk(cudaGetLastError());
}

void test_cuda_transform_optimized(float* data, dim3 data_dims, cudaStream_t stream) {

     // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid((data_dims.x/512)/4);

    fk::binary_operation_scalar<fk::binary_mul<float>, float> op1 = {5.f};
    fk::binary_operation_pointer<fk::binary_sum<float>, float> op2 = {data};
    fk::binary_operation_scalar<fk::binary_div<float>, float> op3 = {2.f};
    fk::binary_operation_scalar<fk::binary_mul<float>, float> op4 = {5.f};
    fk::binary_operation_scalar<fk::binary_div<float>, float> op5 = {3.f};
    fk::binary_operation_scalar<fk::binary_mul<float>, float> op6 = {7.f};

    fk::cuda_transform_optimized<<<grid, thread_block, 0, stream>>>(data_dims.x, data, data, op1, op2, op3, op4, op5, op6);
    gpuErrchk(cudaGetLastError());

}

template <typename T>
bool testPtr_2D() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    constexpr size_t width_crop = 300;
    constexpr size_t height_crop = 200;

    fk::Point startPoint = {100, 200};

    fk::Ptr3D<T> input(width, height);
    fk::Ptr3D<T> cropedInput = input.crop(startPoint, width_crop, height_crop);
    fk::Ptr3D<T> output(width_crop, height_crop);
    fk::Ptr3D<T> outputBig(width, height);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    dim3 block2D(32,8);
    dim3 grid2D(std::ceil(width_crop / (float)block2D.x),
                std::ceil(height_crop / (float)block2D.y));
    dim3 grid2DBig(std::ceil(width / (float)block2D.x),
                   std::ceil(height / (float)block2D.y));

    fk::memory_write_scalar_2D<fk::perthread_write_2D<T>, T> opFinal_2D = { output };
    fk::memory_write_scalar_2D<fk::perthread_write_2D<T>, T> opFinal_2DBig = { outputBig };

    for (int i=0; i<100; i++) {
        fk::cuda_transform_noret_2D<<<grid2D, block2D, 0, stream>>>(cropedInput.d_ptr(), opFinal_2D);
        fk::cuda_transform_noret_2D<<<grid2DBig, block2D, 0, stream>>>(input.d_ptr(), opFinal_2DBig);
    }

    cudaError_t err = cudaStreamSynchronize(stream);

    // TODO: use some values and check results correctness

    if (err != cudaSuccess) {
        return false;
    } else {
        return true;
    }
}

int main() {
    bool test2Dpassed = true;

    test2Dpassed &= testPtr_2D<uchar>();
    test2Dpassed &= testPtr_2D<uchar3>();
    test2Dpassed &= testPtr_2D<float>();
    test2Dpassed &= testPtr_2D<float3>();

    if (test2Dpassed) {
        std::cout << "testPtr_2D Success!!" << std::endl; 
    } else {
        std::cout << "testPtr_2D Failed!!" << std::endl;
    }

    return 0;
}