/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "cuda_vector_utils.h"

namespace fk {

template <typename I, typename O=I>
struct perthread_write {
    __device__ void operator()(I input, O* output) { output[GLOBAL_ID] = input; }
};

template <typename I, typename Enabler=void>
struct perthread_split_write;

template <typename I>
struct perthread_split_write<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 2>> {
    __device__ void operator()(I input, decltype(I::x)* output1, decltype(I::y)* output2) { output1[GLOBAL_ID] = input.x; 
                                                                                            output2[GLOBAL_ID] = input.y; }
};

template <typename I>
struct perthread_split_write<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 3>> {
    __device__ void operator()(I input, 
                               decltype(I::x)* output1, 
                               decltype(I::y)* output2,
                               decltype(I::z)* output3) { output1[GLOBAL_ID] = input.x; 
                                                          output2[GLOBAL_ID] = input.y; 
                                                          output3[GLOBAL_ID] = input.z; }
};

template <typename I>
struct perthread_split_write<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 4>> {
    __device__ void operator()(I input, decltype(I::x)* output1, 
                                        decltype(I::y)* output2,
                                        decltype(I::z)* output3,
                                        decltype(I::w)* output4) { output1[GLOBAL_ID] = input.x; 
                                                                   output2[GLOBAL_ID] = input.y; 
                                                                   output3[GLOBAL_ID] = input.z;
                                                                   output4[GLOBAL_ID] = input.w; }
};

}