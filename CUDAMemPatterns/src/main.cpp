#include <vector>
#include "fast_kernel.h"
#include "cpu_baseline.h"
#include "standard_kernels.h"
#include "cuda_vector_types.h"

#define SIZE 3840*2160

int main() {

    float* data = (float*)malloc(sizeof(float)*SIZE);
    float* cpu_o_data = (float*)malloc(sizeof(float)*SIZE);
    float* h_o_fast_data = (float*)malloc(sizeof(float)*SIZE);
    float* h_o_data = (float*)malloc(sizeof(float)*SIZE);

    for (int i=0; i<SIZE; ++i) {
        data[i] = (float)(i%100);
    }

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    float *d_data, *d_o_data;
    gpuErrchk(cudaMalloc(&d_data, sizeof(float)*SIZE));
    gpuErrchk(cudaMalloc(&d_o_data, sizeof(float)*SIZE));
    gpuErrchk(cudaMemcpyAsync(d_data, data, sizeof(float)*SIZE, cudaMemcpyHostToDevice, stream));

    dim3 size(SIZE);
    test_mult_sum_div_float_standard(d_data, d_o_data, size, stream);
    // Faster for small arrays
    //test_mult_sum_div_float(d_data, size, stream);
    // Faster for big arrays
    test_cuda_transform_optimized(d_data, size, stream);

    gpuErrchk(cudaMemcpyAsync(h_o_fast_data, d_data, sizeof(float)*SIZE, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(h_o_data, d_o_data, sizeof(float)*SIZE, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    cpu_binary_operation<float> op1 = {scalar, cpu_binary_mul<float>, 5};
    cpu_binary_operation<float> op2 = {pointer, cpu_binary_sum<float>, 0, data};
    cpu_binary_operation<float> op3 = {scalar, cpu_binary_div<float>, 2};
    cpu_binary_operation<float> op4 = {scalar, cpu_binary_mul<float>, 5};
    cpu_binary_operation<float> op5 = {scalar, cpu_binary_div<float>, 3};
    cpu_binary_operation<float> op6 = {scalar, cpu_binary_mul<float>, 7};

    cpu_cuda_transform(data, cpu_o_data, size, op1, op2, op3, op4, op5, op6);

    std::cout << "Executed!!" << std::endl;
    bool success = true;
    for (int i=0; i<SIZE; ++i) {
        if (success) success = 0.001 > abs(cpu_o_data[i] - h_o_fast_data[i]);
        if (success) success = 0.001 > abs(cpu_o_data[i] - h_o_data[i]);
        //std::cout << "cpu_o_data " << cpu_o_data[i] << " == " << h_o_fast_data[i] << " h_o_fast_data " << h_o_data[i] << " h_o_data" << std::endl;
    }

    free(data);
    free(cpu_o_data);
    free(h_o_fast_data);
    free(h_o_data);

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_o_data));

    if (success) {
        std::cout << "Success!!" << std::endl;
    } else {
        std::cout << "Fail!!" << std::endl;
    }

    gpuErrchk(cudaStreamDestroy(stream));

    float3 var_test = make_<float3>(255, 255, 255);
    std::cout << "The values of var_test are " << var_test.x << ", " << var_test.y << ", " << var_test.z << std::endl;

    return 0;
}
