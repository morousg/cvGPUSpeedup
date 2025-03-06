/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#ifdef WIN32
#include <intellisense/main.h>
#endif

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>
#include <fused_kernel/fused_kernel.cuh>

using PtrToTest = fk::Ptr2D<uchar3>;
constexpr int WIDTH = 64;
constexpr int HEIGHT = 64;

PtrToTest test_return_by_value() {
    return PtrToTest(WIDTH, HEIGHT);
}

const PtrToTest& test_return_by_const_reference(const PtrToTest& somePtr) {
    return somePtr;
}

PtrToTest& test_return_by_reference(PtrToTest& somePtr) {
    return somePtr;
}

int launch() {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    PtrToTest test0(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);
    fk::setTo(fk::make_<uchar3>(1, 2, 3), test0);
    bool h_correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool3 boolVect = *fk::PtrAccessor<fk::_2D>::cr_point(fk::Point(x, y), test0.ptr()) == fk::make_<uchar3>(1, 2, 3);
            h_correct &= fk::VectorAnd<bool3>::exec(boolVect);
        }
    }

    PtrToTest test1(WIDTH, HEIGHT);

    auto test2 = PtrToTest(WIDTH, HEIGHT);

    PtrToTest test3;
    test3 = PtrToTest(WIDTH, HEIGHT);

    auto test4 = test_return_by_value();
    PtrToTest somePtr(WIDTH, HEIGHT);
    const PtrToTest& test5 = test_return_by_const_reference(somePtr);
    PtrToTest& test6 = test_return_by_reference(somePtr);

    bool result = test1.getRefCount() == 1;
    result &= test2.getRefCount() == 1;
    result &= test3.getRefCount() == 1;
    result &= test4.getRefCount() == 1;
    result &= test5.getRefCount() == 1;
    result &= test6.getRefCount() == 1;

    PtrToTest test7(WIDTH, HEIGHT);
    PtrToTest h_test7(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);
    fk::setTo(fk::make_<uchar3>(3,6,10), test7, stream);
    gpuErrchk(cudaMemcpy2DAsync(h_test7.ptr().data, h_test7.ptr().dims.pitch,
                                test7.ptr().data, test7.ptr().dims.pitch,
                                WIDTH * sizeof(uchar3), HEIGHT, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool h_correct2{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool3 boolVect = *fk::PtrAccessor<fk::_2D>::cr_point(fk::Point(x, y), h_test7.ptr()) == fk::make_<uchar3>(3, 6, 10);
            h_correct2 &= fk::VectorAnd<bool3>::exec(boolVect);
        }
    }

    return result && h_correct && h_correct2 ? 0 : -1;
}
