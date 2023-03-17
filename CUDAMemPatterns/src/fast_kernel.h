#ifndef CUDAMEMPATTERNS_SRC_KERNEL_H_
#define CUDAMEMPATTERNS_SRC_KERNEL_H_

#include "utils.h"

template <typename Operator, typename I1, typename I2, typename O>
struct MY_ALIGN(16) _binary_operation_scalar {
    I2 scalar;
    Operator nv_operator;
};

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_scalar = typename _binary_operation_scalar<Operator, I1, I2, O>;

template <typename Operator, typename I1, typename I2, typename O>
struct MY_ALIGN(16) _binary_operation_pointer {
    I2* pointer;
    Operator nv_operator;
    I2 temp_register[4];
};

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_pointer = typename _binary_operation_pointer<Operator, I1, I2, O>;

void test_mult_sum_div_float(float* data, dim3 data_dims, cudaStream_t stream);

void test_cuda_transform_optimized(float* data, dim3 data_dims, cudaStream_t stream);

#endif  // CUDAMEMPATTERNS_SRC_KERNEL_H_
