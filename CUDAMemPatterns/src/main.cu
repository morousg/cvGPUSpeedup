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

    constexpr size_t NUM_ELEMENTS = 3;

    gpuErrchk(cudaMalloc(&d_input, sizeof(uchar3) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_x, sizeof(float) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_y, sizeof(float) * NUM_ELEMENTS));
    gpuErrchk(cudaMalloc(&d_out_z, sizeof(float) * NUM_ELEMENTS));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    uchar3 pixel = {0u, 128u, 255u};
    uchar3 h_input[NUM_ELEMENTS];
    float h_out_x[NUM_ELEMENTS], h_out_y[NUM_ELEMENTS], h_out_z[NUM_ELEMENTS];

    for (int i=0; i<NUM_ELEMENTS; i++) {
        h_input[i] = pixel;
    }

    gpuErrchk(cudaMemcpyAsync(d_input, h_input, sizeof(uchar3) * NUM_ELEMENTS, cudaMemcpyHostToDevice, stream));
    
    unary_operation_scalar<unary_cuda_vector_cast<uchar3,float3>, uchar3, float3> op1 = {};
    binary_operation_scalar<binary_sub<float3>, float3, float3> op2 = { make_<float3>(1.f, 1.f, 1.f) };
    binary_operation_scalar<binary_div<float3>, float3, float3> op3 = { make_<float3>(2.f, 2.f, 2.f) };
    split_write_scalar<perthread_split_write<float3>, float3> op4 = { d_out_x, d_out_y, d_out_z };

    cuda_transform_noret<<<1,3,0,stream>>>(NUM_ELEMENTS, d_input, op1, op2, op3, op4);

    gpuErrchk(cudaMemcpyAsync(h_out_x, d_out_x, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(h_out_y, d_out_y, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(h_out_z, d_out_z, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    for (int i=0; i<NUM_ELEMENTS; i++) {
      std::cout << "Result = " << h_out_x[i] << ", " << h_out_y[i] << ", " << h_out_z[i] << std::endl;
    }

    return 0;
}