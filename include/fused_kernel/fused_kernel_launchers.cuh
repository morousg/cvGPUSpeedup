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
    struct SequenceSelectorType {
        FK_HOST_DEVICE_FUSE uint at(const uint& index) {
            if (index > 0) return 2u;
            else return 1u;
        }
    };

    template <typename T, typename Enabler=void>
    struct TempType;

    template <typename T>
    struct TempType<T, std::enable_if_t<cn<T> == 3>> {
        using type = typename VectorType<typename VectorTraits<T>::base, 4>::type;
    };

    template <typename T>
    struct TempType<T, std::enable_if_t<cn<T> != 3>> {
        using type = T;
    };

    template <typename T, int COLOR_PLANES, int BATCH>
    class CircularTensor : public Tensor<T> {

        using SourceT = typename VectorType<T, COLOR_PLANES>::type;

        using ReadDeviceFunctions = TypeList<ReadDeviceFunction<CircularTensorRead<CircularDirection::Descendent, TensorRead<SourceT>, BATCH>>,
                                             ReadDeviceFunction<CircularTensorRead<CircularDirection::Descendent, TensorSplitRead<SourceT>, BATCH>>>;

        using WriteDeviceFunctions = TypeList<WriteDeviceFunction<TensorWrite<SourceT>>,
                                              WriteDeviceFunction<TensorSplitWrite<SourceT>>>;

    public:
        __host__ inline constexpr CircularTensor() {};

        __host__ inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
            Tensor<T>(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_),
            m_tempTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_) {};

        __host__ inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
            this->allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
            m_tempTensor.allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
        }

        template <typename... DeviceFunctionTypes>
        __host__ inline constexpr void update(const cudaStream_t& stream,
                                              const DeviceFunctionTypes&... deviceFunctionInstances) {
            const auto writeDeviceFunction = last(deviceFunctionInstances...);
            using writeDFType = std::decay_t<decltype(writeDeviceFunction)>;
            using writeOpType = typename writeDFType::Operation;
            using equivalentReadDFType = EquivalentType_t<writeDFType, WriteDeviceFunctions, ReadDeviceFunctions>;

            MidWriteDeviceFunction<CircularTensorWrite<CircularDirection::Ascendent, writeOpType, BATCH>> updateWriteToTemp;
            updateWriteToTemp.params.first = m_nextUpdateIdx;
            updateWriteToTemp.params.params = m_tempTensor.ptr();

            const auto updateOps = buildOperationSequence_tup(insert_before_last(updateWriteToTemp, deviceFunctionInstances...));

            // Build copy pipeline
            equivalentReadDFType nonUpdateRead;
            nonUpdateRead.params.first = m_nextUpdateIdx;
            nonUpdateRead.params.params = m_tempTensor.ptr();
            nonUpdateRead.activeThreads.x = this->ptr_a.dims.width;
            nonUpdateRead.activeThreads.y = this->ptr_a.dims.height;
            nonUpdateRead.activeThreads.z = BATCH;

            const auto copyOps = buildOperationSequence(nonUpdateRead, writeDeviceFunction);

            dim3 grid((uint)ceil((float)this->ptr_a.dims.width / (float)this->adjusted_blockSize.x),
                      (uint)ceil((float)this->ptr_a.dims.height / (float)this->adjusted_blockSize.y),
                      BATCH);

            cuda_transform_divergent_batch<SequenceSelectorType> << <grid, this->adjusted_blockSize, 0, stream >> > (updateOps, copyOps);
           
            m_nextUpdateIdx = (m_nextUpdateIdx + 1) % BATCH;
            gpuErrchk(cudaGetLastError());
        }
        
    private:
        Tensor<T> m_tempTensor;
        int m_nextUpdateIdx{0};
    };

};
