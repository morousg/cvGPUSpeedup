#ifndef CUDAMEMPATTERNS_SRC_CPU_BASELINE_H_
#define CUDAMEMPATTERNS_SRC_CPU_BASELINE_H_

#include <functional>
#include "utils.h"

template <typename I1, typename I2=I1, typename O=I1>
O cpu_binary_sum(I1 input_1, I2 input_2) {
    return input_1 + input_2;
}

template <typename I1, typename I2=I1, typename O=I1>
O cpu_binary_mul(I1 input_1, I2 input_2) {
    return input_1 * input_2;
}

template <typename I1, typename I2=I1, typename O=I1>
O cpu_binary_div(I1 input_1, I2 input_2) {
    return input_1 / input_2;
}

enum transform_patern {
    scalar,
    pointer
};

template <typename I1, typename I2, typename O>
struct _binary_operation {
    transform_patern parameter;
    std::function<O(I1,I2)> nv_operator;
    I2 scalar;
    I2* pointer;
    I2 temp_register[4];
};

template <typename I1, typename I2=I1, typename O=I1>
using cpu_binary_operation = typename _binary_operation<I1, I2, O>;

template <typename O>
O operate(int i, O i_data){
    return i_data;
}

template <typename I, typename O, typename I2, typename... operations>
O operate(int i, I i_data, cpu_binary_operation<I, I2, O> op, operations... ops){

    if (op.parameter == scalar) {
        O temp = op.nv_operator(i_data, op.scalar);
        return operate(i, temp, ops...);
    } else {
        O temp = op.nv_operator(i_data, op.pointer[i]);
        return operate(i, temp, ops...);
    }

}

template<typename I, typename O, typename... operations>
void cpu_cuda_transform(I* i_data, O* o_data, dim3 size, operations... ops) {
    for (int i=0; i<size.x; ++i) {
        o_data[i] = operate(i, i_data[i], ops...);
    }
}

#endif // CUDAMEMPATTERNS_SRC_CPU_BASELINE_H_
