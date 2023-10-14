/* 
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
    using Type = T;
    using ParamsType = RawPtr<D, T>;
    FK_DEVICE_FUSE Type exec(const Point& thread, const ParamsType& ptr) {
        return *PtrAccessor<D>::cr_point(thread, ptr);
    }
};

template <ND D, typename T>
struct PerThreadWrite {
    using Type = T;
    using ParamsType = RawPtr<D, T>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& output) {
        *PtrAccessor<D>::point(thread, output) = input;
    }
};

template <typename T>
struct TensorRead {
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
    FK_DEVICE_FUSE Type exec(const Point& thread, const ParamsType& ptr) {
        return *PtrAccessor<_3D>::cr_point(thread, ptr);
    }
};

template <typename T>
struct TensorWrite {
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& output) {
        *PtrAccessor<_3D>::point(thread, output) = input;
    }
};


template <typename T>
struct TensorSplitWrite {
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& ptr) {
        static_assert(cn<Type> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        using OutputType = typename VectorTraits<Type>::base;
        OutputType* const work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        *(work_plane + planePixels) = input.y;
        if constexpr (cn<Type> >= 3) {
            *(work_plane + (planePixels * 2)) = input.z;
        }
        if constexpr (cn<Type> == 4) {
            *(work_plane + (planePixels * 3)) = input.w;
        }
    }
};

template <typename T>
struct TensorTSplitWrite {
    using Type = T;
    using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& ptr) {
        static_assert(cn<Type> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

        using OutputType = typename VectorTraits<Type>::base;
        *PtrAccessor<T3D>::point(thread, ptr, 0) = input.x;
        *PtrAccessor<T3D>::point(thread, ptr, 1) = input.y;
        if constexpr (cn<Type> >= 3) {
            *PtrAccessor<T3D>::point(thread, ptr, 2) = input.z;
        }
        if constexpr (cn<Type> == 4) {
            *PtrAccessor<T3D>::point(thread, ptr, 3) = input.w;
        }
    }
};

template <typename T>
struct TensorSplitRead {
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
    FK_DEVICE_FUSE Type exec(const Point& thread, const ParamsType& ptr) {
        static_assert(cn<Type> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        using OutputType = typename VectorTraits<Type>::base;
        const OutputType* const work_plane = PtrAccessor<_3D>::cr_point(thread, ptr);
        if constexpr (cn<Type> == 2) {
            return make_<Type>(*work_plane, *(work_plane + planePixels));
        } else if constexpr (cn<Type> == 3) {
            return make_<Type>(*work_plane, *(work_plane + planePixels),
                               *(work_plane + (planePixels * 2)));
        } else {
            return make_<Type>(*work_plane,
                               *(work_plane + planePixels),
                               *(work_plane + (planePixels * 2)),
                               *(work_plane + (planePixels * 3)));
        }
    }
};

template <typename T>
struct TensorTSplitRead {
    using Type = T;
    using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
    FK_DEVICE_FUSE Type exec(const Point& thread, const ParamsType& ptr) {
        static_assert(cn<Type> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

        using OutputType = typename VectorTraits<Type>::base;
        const OutputType x = *PtrAccessor<T3D>::cr_point(thread, ptr, 0);
        if constexpr (cn<Type> == 2) {
            const OutputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
            return make_<Type>(x, y);
        } else if constexpr (cn<Type> == 3) {
            const OutputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
            const OutputType z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
            return make_<Type>(x, y, z);
        } else {
            const OutputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
            const OutputType z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
            const OutputType w = *PtrAccessor<T3D>::cr_point(thread, ptr, 3);
            return make_<Type>(x, y, z, w);
        }
    }
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
    using Type = T;
    using ParamsType = SplitWriteParams<D, T>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& params) {
        static_assert(cn<Type> >= 2, "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
        *PtrAccessor<D>::point(thread, params.x) = input.x;
        *PtrAccessor<D>::point(thread, params.y) = input.y;
        if constexpr (cn<Type> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
        if constexpr (cn<Type> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
    }
};

template <typename Operation, int NPtr>
struct BatchRead {
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
    FK_DEVICE_FUSE const Type exec(const Point& thread, const typename Operation::ParamsType (&params)[NPtr]) {
        return Operation::exec(thread, params[thread.z]);
    }
};

template <typename Operation, int NPtr>
struct BatchWrite {
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const typename Operation::ParamsType (&params)[NPtr]) {
        Operation::exec(thread, input, params[thread.z]);
    }
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

template <PixelFormat PF>
struct ReadYUV {
    using Type = VectorType_t<YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>, PixelFormatTraits<PF>::cn>;
    using ParamsType = RawPtr<_2D, YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>>;
    static __device__ __forceinline__ const Type exec(const Point& thread, const ParamsType& params) {
        if constexpr (PF == NV12 || PF == P010 || PF == P016 || PF == P210 || PF == P216) {
            // Planar luma
            const YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth> Y = *PtrAccessor<_2D>::cr_point(thread, params);

            // Packed chroma
            const PtrDims<_2D> dims = params.dims;
            const RawPtr<_2D, VectorType_t<YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>, 2>> chromaPlane{
                (VectorType_t<YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>, 2>*)((uchar*)params.data + dims.pitch * dims.height),
                { dims.width >> 1, dims.height >> 1, dims.pitch }
            };
            const ColorSpace CS = (ColorSpace)PixelFormatTraits<PF>::space;
            const VectorType_t<YUVChannelType<(ColorDepth)PixelFormatTraits<PF>::depth>, 2> UV =
                *PtrAccessor<_2D>::cr_point({thread.x >> 1, CS == YUV420 ? thread.y >> 1 : thread.y, thread.z}, chromaPlane);

            return { Y, UV.x, UV.y };
        } else if constexpr (PF == NV21) {
            // Planar luma
            const uchar Y = *PtrAccessor<_2D>::cr_point(thread, params);

            // Packed chroma
            const PtrDims<_2D> dims = params.dims;
            const RawPtr<_2D, uchar2> chromaPlane{
                (uchar2*)((uchar*)params.data + dims.pitch * dims.height), { dims.width >> 1, dims.height >> 1, dims.pitch }
            };
            const uchar2 VU = *PtrAccessor<_2D>::cr_point({ thread.x >> 1, thread.y >> 1, thread.z }, chromaPlane);

            return { Y, VU.y, VU.x };
        } else if constexpr (PF == Y216 || PF == Y210) {
            const PtrDims<_2D> dims = params.dims;
            const RawPtr<_2D, ushort4> image{ (ushort4*)params.data, {dims.width >> 1, dims.height, dims.pitch} };
            const ushort4 pixel = *PtrAccessor<_2D>::cr_point({thread.x >> 1, thread.y, thread.z}, image);
            const bool isEvenThread = UnaryIsEven<uint>::exec(thread.x);

            return { isEvenThread ? pixel.x : pixel.z, pixel.y, pixel.w };
        } else if constexpr (PF == Y416) {
            // AVYU
            // We use ushort as the type, to be compatible with the rest of the cases
            const RawPtr<_2D, ushort4> readImage{params.data, params.dims};
            const ushort4 pixel = *PtrAccessor<_2D>::cr_point(thread, params);
            return { pixel.z, pixel.w, pixel.y, pixel.x };
        }
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
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
    FK_DEVICE_FUSE const Type exec(const Point& thread, const ParamsType& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
    }
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularBatchWrite {
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
    }
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularTensorRead {
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
    FK_DEVICE_FUSE const Type exec(const Point& thread, const ParamsType& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        return Operation::exec(newThreadIdx, c_params.params);
    }
};

template <CircularDirection direction, typename Operation, int BATCH>
struct CircularTensorWrite {
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
    FK_DEVICE_FUSE void exec(const Point& thread, const Type& input, const ParamsType& c_params) {
        const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
        Operation::exec(newThreadIdx, input, c_params.params);
    }
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
    static __device__ __forceinline__ const Type exec(const Point& thread, const ParamsType& params) {
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
