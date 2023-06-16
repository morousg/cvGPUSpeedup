/*Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.*/

#include "fused_kernel.cuh"

namespace fk {

template <typename T, int BATCH>
using ReadOperations = TypeList<ReadDeviceFunction<CircularTensorRead<TensorRead<T>, BATCH>>,
                                ReadDeviceFunction<CircularTensorRead<TensorSplitRead<T>, BATCH>>;

template <typename T, int BATCH>
using WriteOperations = TypeList<WriteDeviceFunction<CircularTensorWrite<TensorWrite<T>, BATCH>>,
                                 WriteDeviceFunction<CircularTensorWrite<TensorSplitWrite<T>, BATCH>>;

template <typename WriteOperation, typename T, int BATCH>
struct equivalenReadOperation {
    using type = TypeFromIndex_t<TypeIndex_v<WriteOperation, WriteOperations<T, BATCH>>, ReadOperations<T, BATCH>>;
};

template <typename I, typename O, int BATCH, typename... UpdateOperations>
inline constexpr void update_circular_batch(const Ptr2D<I>& newPlane, const int& newPlaneIdx, const Tensor<O>& previousOutput,
                                            const Tensor<O>& newOutput, UpdateOperations... ops) {
    auto writeOperation = last(ops...);
    using writeType = decltype(writeOperation);
    using equivalentReadType = typename equivalenReadOperation<decltype(writeOperation), O, BATCH>::type;
     
    OperationSequence<UpdateOperations...> ops1 = buildOperationSequence(ops...);
    OperationSequence<equivalentReadType, writeType> ops2 = buildOperationSequence(equivalentReadType{previousOutput},);
}
}
