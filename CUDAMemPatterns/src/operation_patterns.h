#pragma once
#include "cuda_utils.h"

template <typename Operator, typename I1, typename I2, typename O>
struct MY_ALIGN(16) _binary_operation_scalar {
    I2 scalar;
    Operator nv_operator;
};
template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_scalar = _binary_operation_scalar<Operator, I1, I2, O>;

template <typename Operator, typename I1, typename I2, typename O>
struct MY_ALIGN(16) _binary_operation_pointer {
    I2* pointer;
    Operator nv_operator;
    I2 temp_register[4];
};
template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_pointer = _binary_operation_pointer<Operator, I1, I2, O>;

template <typename Operator, typename I, typename O>
struct MY_ALIGN(16) _unary_operation_scalar {
    Operator nv_operator;
};
template <typename Operator, typename I, typename O>
using unary_operation_scalar = _unary_operation_scalar<Operator, I, O>;