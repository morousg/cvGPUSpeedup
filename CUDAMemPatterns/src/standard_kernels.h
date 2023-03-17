#ifndef CUDAMEMPATTERNS_SRC_STANDARD_KERNELS_H_
#define CUDAMEMPATTERNS_SRC_STANDARD_KERNELS_H_

#include "utils.h"

void test_mult_sum_div_float_standard(float* i_data, float* o_data, dim3 data_dims, cudaStream_t stream);

#endif //CUDAMEMPATTERNS_SRC_STANDARD_KERNELS_H_
