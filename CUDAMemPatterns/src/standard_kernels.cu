#include "standard_kernels.h"

template<typename I, typename I2, typename O, typename operation>
__global__ void cuda_transform_one(int size, I* i_data, O* o_data, operation op, I2 parameter) {
    if (GLOBAL_ID < size) o_data[GLOBAL_ID] = op(i_data[GLOBAL_ID], parameter);
}

template<typename I, typename I2, typename O, typename operation>
__global__ void cuda_transform_one(int size, I* i_data, O* o_data, operation op, I2* parameter) {
    if (GLOBAL_ID < size) o_data[GLOBAL_ID] = op(i_data[GLOBAL_ID], parameter[GLOBAL_ID]);
}

void test_mult_sum_div_float_standard(float* i_data, float* o_data, dim3 data_dims, cudaStream_t stream) {
    // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid(data_dims.x/512);

    binary_mul<float> op1;
    binary_sum<float> op2;
    binary_div<float> op3;
    binary_mul<float> op4;
    binary_div<float> op5;
    binary_mul<float> op6;

    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, i_data, o_data, op1, 5.f);
    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, o_data, o_data, op2, i_data);
    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, o_data, o_data, op3, 2.f);
    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, o_data, o_data, op4, 5.f);
    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, o_data, o_data, op5, 3.f);
    cuda_transform_one<<<grid, thread_block, 0, stream>>>(data_dims.x, o_data, o_data, op6, 7.f);
    gpuErrchk(cudaGetLastError());
}
