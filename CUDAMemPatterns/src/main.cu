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

#include "fast_kernel.h"

void test_mult_sum_div_float(float* data, dim3 data_dims, cudaStream_t stream) {
    // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid(data_dims.x/512);

    binary_operation_scalar<binary_mul<float>, float> op1 = {5.f};
    binary_operation_pointer<binary_sum<float>, float> op2 = {data};
    binary_operation_scalar<binary_div<float>, float> op3 = {2.f};
    binary_operation_scalar<binary_mul<float>, float> op4 = {5.f};
    binary_operation_scalar<binary_div<float>, float> op5 = {3.f};
    binary_operation_scalar<binary_mul<float>, float> op6 = {7.f};

    cuda_transform<<<grid, thread_block, 0, stream>>>(data_dims.x, data, data, op1, op2, op3, op4, op5, op6);
    gpuErrchk(cudaGetLastError());
}

void test_cuda_transform_optimized(float* data, dim3 data_dims, cudaStream_t stream) {

     // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid((data_dims.x/512)/4);

    binary_operation_scalar<binary_mul<float>, float> op1 = {5.f};
    binary_operation_pointer<binary_sum<float>, float> op2 = {data};
    binary_operation_scalar<binary_div<float>, float> op3 = {2.f};
    binary_operation_scalar<binary_mul<float>, float> op4 = {5.f};
    binary_operation_scalar<binary_div<float>, float> op5 = {3.f};
    binary_operation_scalar<binary_mul<float>, float> op6 = {7.f};

    cuda_transform_optimized<<<grid, thread_block, 0, stream>>>(data_dims.x, data, data, op1, op2, op3, op4, op5, op6);
    gpuErrchk(cudaGetLastError());

}

int main() {
    uchar3* d_input;
    float* d_out_x;
    float* d_out_y;
    float* d_out_z;

    constexpr size_t NUM_ELEMENTS = 8192;

    std::cout << "Before any cuda call" << std::endl;

    gpuErrchk(cudaMalloc(&d_input, sizeof(uchar3) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_x, sizeof(float) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_y, sizeof(float) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_z, sizeof(float) * NUM_ELEMENTS));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    uchar3 pixel = { 0u, 128u, 255u };
    uchar3* h_input = (uchar3*)malloc(sizeof(uchar3)*NUM_ELEMENTS);
    float* h_out_x = (float*)malloc(sizeof(float)*NUM_ELEMENTS);
    float* h_out_y = (float*)malloc(sizeof(float)*NUM_ELEMENTS);
    float* h_out_z = (float*)malloc(sizeof(float)*NUM_ELEMENTS);

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_input[i] = pixel;
    }

    gpuErrchk(cudaMemcpyAsync(d_input, h_input, sizeof(uchar3) * NUM_ELEMENTS, cudaMemcpyHostToDevice, stream));

    unary_operation_scalar<unary_cuda_vector_cast<uchar3, float3>, uchar3, float3> op1 = {};
    binary_operation_scalar<binary_sub<float3>, float3, float3> op2 = { make_<float3>(1.f, 1.f, 1.f) };
    binary_operation_scalar<binary_div<float3>, float3, float3> op3 = { make_<float3>(2.f, 2.f, 2.f) };
    split_write_scalar<perthread_split_write<float3>, float3> op4 = { d_out_x, d_out_y, d_out_z };

    std::cout << "Before kernel" << std::endl;

    dim3 block(256);
    dim3 grid(NUM_ELEMENTS / 256);
    for (int i = 0; i < 1000000; i++) {
        cuda_transform_noret << <grid, block, 0, stream >> > (NUM_ELEMENTS, d_input, op1, op2, op3, op4);
    }

    std::cout << "After kernel" << std::endl;

    gpuErrchk(cudaMemcpyAsync(h_out_x, d_out_x, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(h_out_y, d_out_y, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(h_out_z, d_out_z, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));

    std::cout << "After copies" << std::endl;

    gpuErrchk(cudaStreamSynchronize(stream));

    std::cout << "After sync" << std::endl;

    //for (int i=0; i<NUM_ELEMENTS; i++) {
    std::cout << "Result = " << h_out_x[0] << ", " << h_out_y[0] << ", " << h_out_z[0] << std::endl;
    //}

    return 0;
}