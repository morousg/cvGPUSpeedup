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
    struct OpSelectorType {
        FK_HOST_DEVICE_FUSE uint at(const uint& index) {
            if (index > 0) return 2u;
            else return 1u;
        }
    };

    template <typename T, int COLOR_PLANES, int BATCH>
    class CircularTensor : public Tensor<T> {

        using SourceT = typename VectorType<T, COLOR_PLANES>::type;

        using ReadDeviceFunctions = TypeList<ReadDeviceFunction<CircularTensorRead<TensorRead<SourceT>, BATCH>>,
                                             ReadDeviceFunction<CircularTensorRead<TensorSplitRead<SourceT>, BATCH>>>;

        using WriteDeviceFunctions = TypeList<WriteDeviceFunction<TensorWrite<SourceT>>,
                                              WriteDeviceFunction<TensorSplitWrite<SourceT>>>;

    public:
        __host__ inline constexpr CircularTensor() {};

        __host__ inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
            Tensor<T>(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_),
            m_tempTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_) {};

        __host__ inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
            allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
            m_tempTensor.allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
        }

        template <typename... Operations>
        __host__ inline constexpr void update(const cudaStream_t& stream, const Operations&... ops) {
            const auto writeOp = last(ops...);
            // assert writeOp is one of TesorWrite or TesorSplitWrite
            using writeDFType = std::decay_t<decltype(writeOp)>;
            using writeOpType = typename writeDFType::Op;
            using equivalentReadDFType = EquivalentType_t<writeDFType, WriteDeviceFunctions, ReadDeviceFunctions>;

            MidWriteDeviceFunction<CircularTensorWrite<writeOpType, BATCH>> updateWriteToTemp;
            updateWriteToTemp.params.first = m_nextUpdateIdx;
            updateWriteToTemp.params.params = m_tempTensor.ptr();

            const auto updateOps = buildOperationSequence_tup(insert_before_last(updateWriteToTemp, ops...));

            equivalentReadDFType nonUpdateRead;
            nonUpdateRead.params.first = m_nextUpdateIdx;
            nonUpdateRead.params.params = m_tempTensor.ptr();
            nonUpdateRead.activeThreads.x = ptr_a.dims.width;
            nonUpdateRead.activeThreads.y = ptr_a.dims.height;
            nonUpdateRead.activeThreads.z = BATCH;

            const auto copyOps = buildOperationSequence(nonUpdateRead, writeOp);

            dim3 grid((uint)ceil((float)ptr_a.dims.width / (float)adjusted_blockSize.x),
                (uint)ceil((float)ptr_a.dims.height / (float)adjusted_blockSize.y),
                BATCH);
            /*Array<int, BATCH> selectorArr;
            selectorArr.at[0] = 1;
            for (int i = 1; i < BATCH; i++) {
                selectorArr.at[i] = 2;
            }*/
            cuda_transform_divergent_batch<OpSelectorType, BATCH> << <grid, adjusted_blockSize, 0, stream >> >(updateOps, copyOps);
            m_nextUpdateIdx = (m_nextUpdateIdx + 1) % BATCH;
            gpuErrchk(cudaGetLastError());
        }

    private:
        Tensor<T> m_tempTensor;
        int m_nextUpdateIdx{ BATCH - 1 };
    };

};
