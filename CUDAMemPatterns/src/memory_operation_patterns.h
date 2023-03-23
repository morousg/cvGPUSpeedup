#pragma once
#include "cuda_vector_utils.h"

template <typename Operator, typename I, typename O>
struct MY_ALIGN(16) _unary_write_scalar {
    Operator nv_operator;
};
template <typename Operator, typename I, typename O>
using unary_write_scalar = _unary_write_scalar<Operator, I, O>;

template <typename Operator, typename I, typename Enabler=void>
struct MY_ALIGN(16) split_write_scalar {};

template <typename Operator, typename I>
struct MY_ALIGN(16) split_write_scalar<Operator, I, typename std::enable_if<NUM_COMPONENTS(I) == 2>::type> {
    decltype(I::x)* x;
    decltype(I::y)* y;
    Operator nv_operator;
};

template <typename Operator, typename I>
struct MY_ALIGN(16) split_write_scalar<Operator, I, typename std::enable_if<NUM_COMPONENTS(I) == 3>::type> {
    decltype(I::x)* x;
    decltype(I::y)* y;
    decltype(I::z)* z;
    Operator nv_operator;
};

template <typename Operator, typename I>
struct MY_ALIGN(16) split_write_scalar<Operator, I, typename std::enable_if<NUM_COMPONENTS(I) == 4>::type> {
    decltype(I::x)* x;
    decltype(I::y)* y;
    decltype(I::z)* z;
    decltype(I::w)* w;
    Operator nv_operator;
};