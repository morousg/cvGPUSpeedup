/* 
   Some device functions are modifications of other libraries

   Copyright 2023 Oscar Amoros Huguet

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
#include "cuda_vector_utils.cuh"
#include "operations.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace fk {

template <ND D, typename T>
struct PerThreadRead {
    FK_DEVICE_FUSE T exec(const Point& thread, const RawPtr<D,T>& ptr) {
        return *PtrAccessor<D>::cr_point(thread, ptr);
    }
    using Type = T;
    using ParamsType = RawPtr<D,T>;
};

template <ND D, typename T>
struct PerThreadWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<D,T>& output) {
        *PtrAccessor<D>::point(thread, output) = input;
    }
    using Type = T;
    using ParamsType = RawPtr<D,T>;
};

template <typename T>
struct TensorRead {
    FK_DEVICE_FUSE T exec(const Point& thread, const RawPtr<_3D, T>& ptr) {
        return *PtrAccessor<_3D>::cr_point(thread, ptr);
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
};

template <typename T>
struct TensorWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<_3D, T>& output) {
        *PtrAccessor<_3D>::point(thread, output) = input;
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
};


template <typename T>
struct TensorSplitWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        static_assert(cn<T> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        typename VectorTraits<T>::base* const work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        *(work_plane + planePixels) = input.y;
        if constexpr (cn<T> >= 3) {
            *(work_plane + (planePixels * 2)) = input.z;
        }
        if constexpr (cn<T> == 4) {
            *(work_plane + (planePixels * 3)) = input.w;
        }
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
};

template <typename T>
struct TensorSplitRead {
    FK_DEVICE_FUSE T exec(const Point& thread,
                          const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        static_assert(cn<T> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        const typename VectorTraits<T>::base* const work_plane = PtrAccessor<_3D>::cr_point(thread, ptr);
        if constexpr (cn<T> == 2) {
            return make_<T>(*work_plane, *(work_plane + planePixels));
        } else if constexpr (cn<T> == 3) {
            return make_<T>(*work_plane, *(work_plane + planePixels),
                            *(work_plane + (planePixels * 2)));
        } else {
            return make_<T>(*work_plane,
                            *(work_plane + planePixels),
                            *(work_plane + (planePixels * 2)),
                            *(work_plane + (planePixels * 3)));
        }
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
};

template <ND D, typename T, typename Enabler=void>
struct SplitWriteParams {};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 2>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 3>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 4>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
    RawPtr<D, decltype(T::w)> w;
};

template <ND D, typename T>
struct SplitWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const SplitWriteParams<D, T>& params) {
        static_assert(cn<T> >= 2, "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
        *PtrAccessor<D>::point(thread, params.x) = input.x;
        *PtrAccessor<D>::point(thread, params.y) = input.y;
        if constexpr (cn<T> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
        if constexpr (cn<T> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
    }
    using Type = T;
    using ParamsType = SplitWriteParams<D, T>;
};

template <typename Operation, int NPtr>
struct BatchRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
                                                       const typename Operation::ParamsType (&params)[NPtr]) {
        return Operation::exec(thread, params[thread.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
};

template <typename Operation, int NPtr>
struct BatchWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
                             const typename Operation::ParamsType (&params)[NPtr]) {
        Operation::exec(thread, input, params[thread.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
};

template <typename I>
struct ResizeReadParams {
    RawPtr<_2D, I> ptr;
    float fx;
    float fy;
};

template <typename I, InterpolationType INTER_T>
struct ResizeRead {
    using Type = typename VectorType<float, cn<I>>::type;
    using ParamsType = ResizeReadParams<I>;
    using InterpolationOp = BinaryInterpolate<I, INTER_T>;

    static __device__ __forceinline__ const Type exec(const Point& thread, const ParamsType& params) {
        // This is what makes the interpolation a resize operation
        const float src_x = thread.x * params.fx;
        const float src_y = thread.y * params.fy;

        static_assert(std::is_same_v<typename InterpolationOp::InputType, float2>, "Wrong InputType for interpolation operation.");
        return InterpolationOp::exec(make_<float2>(src_x, src_y), params.ptr);
    }
};

/* The following code has the following copy right
 
   Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huget)
   Copyright 2023 Mediaproduccion S.L.U. (Guillermo Oyarzun Altamirano)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

enum CircularDirection { Ascendent, Descendent };

template <typename ParamsType>
struct CircularMemoryParams {
    uint first;
    ParamsType params;
};

template <CircularDirection direction, int BATCH>
FK_HOST_DEVICE_CNST Point computeCircularThreadIdx(const Point& currentIdx, const uint& fst) {
    if constexpr (direction == CircularDirection::Ascendent) {
        const uint z = currentIdx.z + fst;
        return { currentIdx.x, currentIdx.y, z >= BATCH ? z - BATCH : z };
    } else {
        const int z = fst - currentIdx.z;
        return { currentIdx.x, currentIdx.y, z < 0 ? static_cast<uint>(BATCH + z) : static_cast<uint>(z) };
    }
}

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularBatchRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
                                                       const CircularMemoryParams<typename Operation::ParamsType[BATCH]>& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularBatchWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
                             const CircularMemoryParams<typename Operation::ParamsType[BATCH]>& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularTensorRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
                                                       const CircularMemoryParams<typename Operation::ParamsType>& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        return Operation::exec(newThreadIdx, c_params.params);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularTensorWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
                             const CircularMemoryParams<typename Operation::ParamsType>& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        Operation::exec(newThreadIdx, input, c_params.params);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
};

enum ROI { OFFSET_THREADS, KEEP_THREAD_IDX };

template <typename Operation>
struct ApplyROIParams {
    int x1, y1; // Top left
    int x2, y2; // Bottom right
    typename Operation::Type defaultValue{};
    typename Operation::ParamsType params;
};

template <typename Operation, ROI USE>
struct ApplyROI {
    using Type = typename Operation::Type;
    using ParamsType = ApplyROIParams<Operation>;

    static __device__ __forceinline__ const typename Type exec(const Point& thread, const ParamsType& params) {
        if (thread.x >= params.x1  && thread.x <= params.x2 && thread.y >= params.y1 && thread.y <= params.y2) {
            if constexpr (USE == OFFSET_THREADS) {
                const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                return Operation::exec(roiThread, params.params);
            } else {
                return Operation::exec(thread, params.params);
            }
        } else {
            return params.defaultValue;
        }
    }
};
}
