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

#include <iostream>

#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/core/utils/template_operations.h>
#include "tests/main.h"

namespace fk {
    template <typename T>
    concept BasicType = (one_of_v<T, StandardTypes>);

    static_assert(BasicType<float>, "Error in test of BasicType");
    static_assert(!BasicType<uchar4>, "Error in test of BasicType");

    template <typename... Types>
    concept AllBasicTypes = (one_of_v<Types, StandardTypes> && ...);

    static_assert(AllBasicTypes<int, float, char, uchar>, "Error in test of AllBasicTypes");
    static_assert(!AllBasicTypes<int, float, float1, uchar>, "Error in test of AllBasicTypes");

    template <typename TypesToFind>
    concept AllBasicTypesInList = (one_of_and_v<TypesToFind, StandardTypes>);

    static_assert(AllBasicTypesInList<StandardTypes>, "Error in test of AllBasicTypesInList");
    static_assert(!AllBasicTypesInList<VAll>, "Error in test of AllBasicTypesInList");
    static_assert(AllBasicTypesInList<TypeList<float, int>>, "Error in test of AllBasicTypesInList");
    static_assert(!AllBasicTypesInList<TypeList<float, int, uint1>>, "Error in test of AllBasicTypesInList");

    template <typename T>
    concept CUDAType = (one_of_v<T, VAll>);

    static_assert(CUDAType<float3>, "Error in test of CUDAType");
    static_assert(!CUDAType<uchar>, "Error in test of CUDAType");

    template <typename... Types>
    concept AllCUDATypes = ((CUDAType<Types>) && ...);

    static_assert(AllCUDATypes<float3, int2>, "Error in test of AllCUDATypes");
    static_assert(!AllCUDATypes<uchar, int3>, "Error in test of AllCUDATypes");

    template <typename TypesToFind>
    concept AllCUDATypesInList = (one_of_and_v<TypesToFind, VAll>);

    template <typename T>
    concept CUDAType1 = (one_of_v<T, VOne>);

    template <typename... Types>
    concept AllCUDAType1 = (one_of_v<Types, VOne> && ...);

    template <typename TypesToFind>
    concept AllCUDAType1InList = (one_of_and_v<TypesToFind, VOne>);

    template <typename T>
    concept CUDAType2 = (one_of_v<T, VTwo>);

    template <typename... Types>
    concept AllCUDAType2 = (one_of_v<Types, VTwo> && ...);

    template <typename TypesToFind>
    concept AllCUDAType2InList = (one_of_and_v<TypesToFind, VTwo>);

    template <typename T>
    concept CUDAType3 = (one_of_v<T, VThree>);

    template <typename... Types>
    concept AllCUDAType3 = (one_of_v<Types, VThree> && ...);

    template <typename TypesToFind>
    concept AllCUDAType3InList = (one_of_and_v<TypesToFind, VThree>);

    template <typename T>
    concept CUDAType4 = (one_of_v<T, VFour>);

    template <typename... Types>
    concept AllCUDAType4 = (one_of_v<Types, VFour> && ...);

    template <typename TypesToFind>
    concept AllCUDAType4InList = (one_of_and_v<TypesToFind, VFour>);

    template <typename T>
    concept BasicOrCUDAType = (BasicType<T> || CUDAType<T>);

    template <typename... Types>
    concept AllBasicOrCUDAType = ((BasicOrCUDAType<Types>) && ...);

    template <typename TypesToFind>
    concept AllBasicOrCUDATypeInList = (one_of_and_v<TypesToFind, VAll> || one_of_and_v<TypesToFind, StandardTypes>);

    template <typename InputType, typename ParamsType, typename OutputType = InputType>
        requires AllBasicTypes<InputType, ParamsType, OutputType>
    struct AddConcepts {
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
    };

    template <typename Operation, typename InputType, typename ParamsType, typename OutputType = InputType>
        requires BasicOrCUDAType<InputType>&& BasicOrCUDAType<ParamsType>&& BasicOrCUDAType<OutputType>
    struct Something {
        using IOTypes = TypeList<InputType, ParamsType, OutputType>;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            if constexpr (requires { AllBasicOrCUDATypeInList<IOTypes>; }) {
                return input;
            } else if constexpr (requires { AllCUDAType1InList<IOTypes>; }) {
                const auto res_x = Operation::exec(input.x, params.x);
                return fk::make_<OutputType>(res_x);
            } else if constexpr (requires { AllCUDAType2InList<IOTypes>; }) {
                return fk::make_<OutputType>(Operation::exec(input.x, params.x),
                                             Operation::exec(input.y, params.y));
            } else if constexpr (requires { AllCUDAType3InList<IOTypes>; }) {
                return fk::make_<OutputType>(Operation::exec(input.x, params.x),
                                             Operation::exec(input.y, params.y),
                                             Operation::exec(input.z, params.z));
            } else if constexpr (requires { AllCUDAType3InList<IOTypes>; }) {
                return fk::make_<OutputType>(Operation::exec(input.x, params.x),
                                             Operation::exec(input.y, params.y),
                                             Operation::exec(input.z, params.z),
                                             Operation::exec(input.w, params.w));
            }
        }
    };

} // namespace fk

template <typename T>
bool testPtr_2D() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    constexpr size_t width_crop = 300;
    constexpr size_t height_crop = 200;

    fk::Point startPoint = {100, 200};

    fk::Ptr2D<T> input(width, height);
    fk::Ptr2D<T> cropedInput = input.crop(startPoint, fk::PtrDims<fk::_2D>(width_crop, height_crop));
    fk::Ptr2D<T> output(width_crop, height_crop);
    fk::Ptr2D<T> outputBig(width, height);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    dim3 block2D(32,8);
    dim3 grid2D((uint)std::ceil(width_crop / (float)block2D.x),
                (uint)std::ceil(height_crop / (float)block2D.y));
    dim3 grid2DBig((uint)std::ceil(width / (float)block2D.x),
                   (uint)std::ceil(height / (float)block2D.y));

    dim3 gridActiveThreadsCrop(cropedInput.dims().width, cropedInput.dims().height);
    dim3 gridActiveThreads(input.dims().width, input.dims().height);
    fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, T>> readCrop{cropedInput, gridActiveThreadsCrop};
    fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, T>> readFull{input, gridActiveThreads};

    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, T>> opFinal_2D = { output };
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, T>> opFinal_2DBig = { outputBig };

    for (int i=0; i<100; i++) {
        fk::cuda_transform<<<grid2D, block2D, 0, stream>>>(readCrop, opFinal_2D);
        fk::cuda_transform<<<grid2DBig, block2D, 0, stream>>>(readFull, opFinal_2DBig);
    }

    cudaError_t err = cudaStreamSynchronize(stream);

    // TODO: use some values and check results correctness

    if (err != cudaSuccess) {
        return false;
    } else {
        return true;
    }
}

int launch() {
    bool test2Dpassed = true;

    test2Dpassed &= testPtr_2D<uchar>();
    test2Dpassed &= testPtr_2D<uchar3>();
    test2Dpassed &= testPtr_2D<float>();
    test2Dpassed &= testPtr_2D<float3>();

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    fk::Ptr2D<uchar> input(64,64);
    fk::Ptr2D<uint> output(64,64);
    
    dim3 gridActiveThreads(64, 64);
    fk::Read<fk::PerThreadRead<fk::_2D, uchar>> read{ input, gridActiveThreads };
    fk::Unary<fk::SaturateCast<uchar, uint>> cast = {};
    fk::Write<fk::PerThreadWrite<fk::_2D, uint>> write { output };

    fk::cuda_transform<<<dim3(1,8),dim3(64,8),0,stream>>>(read, cast, write);

    fk::OperationTuple<fk::PerThreadRead<fk::_2D, uchar>, fk::SaturateCast<uchar, uint>, fk::PerThreadWrite<fk::_2D, uint>> myTup{};

    fk::OpTupUtils<2>::get_params(myTup);
    constexpr bool test1 = std::is_same_v<fk::tuple_element_t<0, decltype(myTup)>, fk::PerThreadRead<fk::_2D, uchar>>;
    constexpr bool test2 = std::is_same_v<fk::tuple_element_t<1, decltype(myTup)>, fk::SaturateCast<uchar, uint>>;
    constexpr bool test3 = std::is_same_v<fk::tuple_element_t<2, decltype(myTup)>, fk::PerThreadWrite<fk::_2D, uint>>;

    gpuErrchk(cudaStreamSynchronize(stream));

    if (test2Dpassed && fk::and_v<test1, test2, test3>) {
        std::cout << "cuda_transform executed!!" << std::endl; 
    } else {
        std::cout << "cuda_transform failed!!" << std::endl;
    }

    return 0;
}