#include "fast_kernel.h"

template <typename O>
__device__ O operate(O i_data){
    return i_data;
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ O operate(I i_data, binary_operation_scalar<Operation, I, I2, O> op, operations... ops) {
    O temp = op.nv_operator(i_data, op.scalar);
    return operate(temp, ops...);
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ O operate(I i_data, binary_operation_pointer<Operation, I, I2, O> op, operations... ops) {
    // we want to have access to I2 in order to ask for the type size for optimizing
    O temp = op.nv_operator(i_data, op.pointer[GLOBAL_ID]);
    return operate(temp, ops...);
}

template<typename I, typename O, typename... operations>
__global__ void cuda_transform(int size, I* i_data, O* o_data, operations... ops) {
    if (GLOBAL_ID < size) o_data[GLOBAL_ID] = operate(i_data[GLOBAL_ID], ops...);
}

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

// As a first optimization, let's suppose we are always using 4byte types, and we read 4 of them per thread.
// Later on we will play with type sizes and so on.

template <typename O>
__device__ O operate_optimized(int i, O i_data) {
    return i_data;
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ O operate_optimized(int i, I i_data, binary_operation_scalar<Operation, I, I2, O> op, operations... ops) {
    O temp = op.nv_operator(i_data, op.scalar);
    return operate_optimized(i, temp, ops...);
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ O operate_optimized(int i, I i_data, binary_operation_pointer<Operation, I, I2, O> op, operations... ops) {
    // we want to have access to I2 in order to ask for the type size for optimizing
    O temp = op.nv_operator(i_data, op.temp_register[i]);
    return operate_optimized(i, temp, ops...);
}

__device__ void parameter_pointer_read() {}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ void parameter_pointer_read(binary_operation_pointer<Operation, I, I2, O>& op, operations&... ops) {
    uint4* temp = (uint4*)(op.pointer);
    uint4 temp_r = temp[GLOBAL_ID];

    I2 temp0, temp1, temp2, temp3;
    temp0 = *((I2*)(&temp_r.x));
    temp1 = *((I2*)(&temp_r.y));
    temp2 = *((I2*)(&temp_r.z));
    temp3 = *((I2*)(&temp_r.w));

    op.temp_register[0] = temp0;
    op.temp_register[1] = temp1;
    op.temp_register[2] = temp2;
    op.temp_register[3] = temp3;
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ void parameter_pointer_read(binary_operation_scalar<Operation, I, I2, O>& op, operations&... ops) {
    parameter_pointer_read(ops...);
}

template<typename I, typename O, typename... operations>
__global__ void cuda_transform_optimized(int size, I* i_data, O* o_data, operations... ops) {

    if (GLOBAL_ID < size) {

        parameter_pointer_read(ops...);

        uint4* i_temp = (uint4*)(i_data);
        uint4 i_temp_r = i_temp[GLOBAL_ID];

        I i_temp0, i_temp1, i_temp2, i_temp3;
        i_temp0 = *((I*)(&i_temp_r.x));
        i_temp1 = *((I*)(&i_temp_r.y));
        i_temp2 = *((I*)(&i_temp_r.z));
        i_temp3 = *((I*)(&i_temp_r.w));

        O res0 = operate_optimized(0, i_temp0, ops...);
        O res1 = operate_optimized(1, i_temp1, ops...);
        O res2 = operate_optimized(2, i_temp2, ops...);
        O res3 = operate_optimized(3, i_temp3, ops...);

        uint4* o_temp = (uint4*)(o_data);
        o_temp[GLOBAL_ID] = make_uint4(*((uint*)&res0), *((uint*)&res1), *((uint*)&res2), *((uint*)&res3));
    }
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
