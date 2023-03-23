#pragma once
#include "cuda_vector_utils.h"

template <typename I, typename O=I>
struct perthread_write {
    __device__ void operator()(I input, O* output) { output[GLOBAL_ID] = input; }
};

template <typename I, typename Enabler=void>
struct perthread_split_write;

template <typename I>
struct perthread_split_write<I, typename std::enable_if<NUM_COMPONENTS(I) == 2>::type> {
    __device__ void operator()(I input, decltype(I::x)* output1, decltype(I::y)* output2) { output1[GLOBAL_ID] = input.x; 
                                                                                            output2[GLOBAL_ID] = input.y; }
};

template <typename I>
struct perthread_split_write<I, typename std::enable_if<NUM_COMPONENTS(I) == 3>::type> {
    __device__ void operator()(I input, 
                               decltype(I::x)* output1, 
                               decltype(I::y)* output2,
                               decltype(I::z)* output3) { output1[GLOBAL_ID] = input.x; 
                                                          output2[GLOBAL_ID] = input.y; 
                                                          output3[GLOBAL_ID] = input.z; }
};

template <typename I>
struct perthread_split_write<I, typename std::enable_if<NUM_COMPONENTS(I) == 4>::type> {
    __device__ void operator()(I input, decltype(I::x)* output1, 
                                        decltype(I::y)* output2,
                                        decltype(I::z)* output3,
                                        decltype(I::w)* output4) { output1[GLOBAL_ID] = input.x; 
                                                                   output2[GLOBAL_ID] = input.y; 
                                                                   output3[GLOBAL_ID] = input.z;
                                                                   output4[GLOBAL_ID] = input.w; }
};