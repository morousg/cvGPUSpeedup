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
#include "operation_types.cuh"
#include <cooperative_groups.h>
#include <vector>
#include <unordered_map>

namespace cg = cooperative_groups;

namespace fk {

struct Point {
    uint x;
    uint y;
    uint z;  
    __device__ __forceinline__ __host__ Point() : x(0), y(0), z(0) {}
    __device__ __forceinline__ __host__ Point(const uint x_, const uint y_ = 0, const uint z_ = 0) : x(x_), y(y_), z(z_) {}
};

inline constexpr uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
    const uint modX = width % blockDimx;
    const uint modY = height % blockDimy;
    const uint th_disabled_in_X = modX == 0 ? 0 : blockDimx - modX;
    const uint th_disabled_in_Y = modY == 0 ? 0 : blockDimy - modY;
    return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
}

template <uint bxS_t, uint byS_t>
struct computeBestSolution {};

template <uint bxS_t>
struct computeBestSolution<bxS_t, 0> {
    static constexpr void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = bxS_t;
            byS = 0;
            if (minDiscardedThreads == 0) return; 
        }
        computeBestSolution<bxS_t, 1>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
    }
};

template <uint bxS_t>
struct computeBestSolution<bxS_t, 1> {
    static constexpr void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = bxS_t;
            byS = 1;
            if constexpr (bxS_t == 3) return;
            if (minDiscardedThreads == 0) return; 
        }
        computeBestSolution<bxS_t+1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
    }
};

template <>
struct computeBestSolution<3, 1> {
    static constexpr void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = 3;
            byS = 1;
        }
    }
};


inline constexpr dim3 getBlockSize(const uint& width, const uint& height) {
    constexpr uint blockDimX[4]    = { 32, 64, 128, 256  };  // Possible block sizes in the x axis
    constexpr uint blockDimY[2][4] = {{ 8,  4,   2,   1},
                                      { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x
    
    uint minDiscardedThreads = UINT_MAX;
    uint bxS = 0; // from 0 to 3
    uint byS = 0; // from 0 to 1
    
    computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

    return dim3(blockDimX[bxS], blockDimY[byS][bxS]);
}

enum MemType { Device, Host, HostPinned };
enum ND { _1D=1, _2D=2, _3D=3 };

template <ND D>
struct PtrDims;

template <>
struct PtrDims<_1D> {
    uint width;
    uint pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0) {}
    __host__ __device__ __forceinline__
    PtrDims(uint width_, uint pitch_=0) : width(width_), pitch(pitch_) {}
};

template <>
struct PtrDims<_2D> {
    uint width;
    uint height;
    uint pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0) {}
    __host__ __device__ __forceinline__
    PtrDims(uint width_, uint height_, uint pitch_=0) :
                    width(width_), height(height_), pitch(pitch_) {}
};

template <>
struct PtrDims<_3D> {
    uint width;
    uint height;
    uint planes;
    uint color_planes;
    uint pitch;
    uint plane_pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0), plane_pitch(0) {}
    __host__ __device__ __forceinline__ 
    PtrDims(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_=0) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(pitch_), plane_pitch(pitch_ * height_) {}
};

template <ND D>
struct PtrAccessor;

template <>
struct PtrAccessor<_1D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE T*__restrict__ cr_point(const Point& p, const T*__restrict__ data, const PtrDims<_1D>& dims) {
        return data + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const T* data, const PtrDims<_1D>& dims) {
        return data + p.x;
    }
};

template <>
struct PtrAccessor<_2D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE const T*__restrict__ cr_point(const Point& p, const T*__restrict__ data, const PtrDims<_2D>& dims) {
        return (const T*)((const char*)data + (p.y * dims.pitch)) + p.x;
    }

    template <typename T>
    FK_HOST_DEVICE_FUSE const T*__restrict__ cr_point(const uint& x, const uint& y, const T*__restrict__ data, const uint& pitch) {
        return (const T*)((const char*)data + (y * pitch)) + x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const T* data, const PtrDims<_2D>& dims) {
        return (T*)((char*)data + (p.y * dims.pitch)) + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const uint& x, const uint& y, const T*__restrict__ data, const uint& pitch) {
        return (T*)((char*)data + (y * pitch)) + x;
    }
};

template <>
struct PtrAccessor<_3D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE T*__restrict__ cr_point(const Point& p, const T*__restrict__ data, const PtrDims<_3D>& dims) {
        return (const T*)((const char*)data + ((dims.plane_pitch * dims.color_planes * p.z) + (p.y * dims.pitch))) + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const T* data, const PtrDims<_3D>& dims) {
        return (T*)((char*)data + ((dims.plane_pitch * dims.color_planes * p.z) + (p.y * dims.pitch))) + p.x;
    }
};

template <ND D, typename T>
struct RawPtr;

template <typename T>
struct RawPtr<_1D, T> {
    T* data;
    PtrDims<_1D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=VectorTraits<T>::cn};
};

template <typename T>
struct RawPtr<_2D, T> {
    T* data;
    PtrDims<_2D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=VectorTraits<T>::cn};
};

template <typename T>
struct RawPtr<_3D, T> {
    T* data;
    PtrDims<_3D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=VectorTraits<T>::cn};
};

template <ND D, typename T>
struct PtrImpl;

template <typename T>
struct PtrImpl<_1D,T> {
    FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_1D>& dims) {
        return dims.pitch;
    }
    FK_HOST_FUSE uint getNumElements(const PtrDims<_1D>& dims) {
        return dims.width;
    }
    FK_HOST_FUSE void d_malloc(RawPtr<_1D,T>& ptr_a) {
        gpuErrchk(cudaMalloc(&ptr_a.data, sizeof(T) * ptr_a.dims.width));
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
    }
    FK_HOST_FUSE void h_malloc_init(PtrDims<_1D>& dims) {}
    FK_HOST_FUSE dim3 getBlockSize(const PtrDims<_1D>& dims) {
        return fk::getBlockSize(dims.width, 1);
    }
};

template <typename T>
struct PtrImpl<_2D,T> {
    FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_2D>& dims) {
        return dims.pitch * dims.height;
    }
    FK_HOST_FUSE uint getNumElements(const PtrDims<_2D>& dims) {
        return dims.width * dims.height;
    }
    FK_HOST_FUSE void d_malloc(RawPtr<_2D,T>& ptr_a) {
        if (ptr_a.dims.pitch == 0) {
            size_t pitch;
            gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
            ptr_a.dims.pitch = pitch;
        } else {
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_2D,T>::sizeInBytes(ptr_a.dims)));
        }
    }
    FK_HOST_FUSE void h_malloc_init(PtrDims<_2D>& dims) {}
    FK_HOST_FUSE dim3 getBlockSize(const PtrDims<_2D>& dims) {
        return fk::getBlockSize(dims.width, dims.height);
    }
};

template <typename T>
struct PtrImpl<_3D,T> {
    FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_3D>& dims) {
        return dims.pitch * dims.height * dims.planes;
    }
    FK_HOST_FUSE uint getNumElements(const PtrDims<_3D>& dims) {
        return dims.width * dims.height * dims.planes;
    }
    FK_HOST_FUSE void d_malloc(RawPtr<_3D,T>& ptr_a) {
        if (ptr_a.dims.pitch == 0) {
            size_t pitch;
            gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height * ptr_a.dims.planes));
            ptr_a.dims.pitch = pitch;
        } else {
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_3D,T>::sizeInBytes(ptr_a)));
        }
        ptr_a.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.heigth;
    }
    FK_HOST_FUSE void h_malloc_init(PtrDims<_3D>& dims) {
        dims.plane_pitch = dims.pitch * dims.height;
    }
    FK_HOST_FUSE dim3 getBlockSize(const PtrDims<_3D>& dims) {
        return fk::getBlockSize(dims.width, dims.height);
    }
};

template <ND D, typename T, typename At>
class Ptr{
protected:
    struct RefPtr {
        void* ptr;
        int cnt;  
    };
    RefPtr* ref{ nullptr };
    RawPtr<D,T> ptr_a;
    dim3 adjusted_blockSize;
    MemType type;
    int deviceID;

    __host__ inline Ptr(RawPtr<D,T> ptr_a_, RefPtr * ref_, dim3 bs_, MemType type_, int devID) : 
                        ptr_a(ptr_a_), ref(ref_), adjusted_blockSize(bs_), type(type_), deviceID(devID) {}

    __host__ inline constexpr uint sizeInBytes() const {
        return PtrImpl<D,T>::sizeInBytes(ptr_a.dims);
    }
    __host__ inline constexpr uint getNumElements() const {
        return PtrImpl<D,T>::getNumElements(ptr_a.dims);
    }
    
    __host__ virtual void allocDevice() {
        int currentDevice;
        gpuErrchk(cudaGetDevice(&currentDevice));
        gpuErrchk(cudaSetDevice(deviceID));
        PtrImpl<D,T>::d_malloc(ptr_a);
        if (currentDevice != deviceID) {
            gpuErrchk(cudaSetDevice(currentDevice));
        }
    }

    __host__ inline void allocHost() {
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width; //Here we don't support padding
        ptr_a.data = (T*)malloc(PtrImpl<D,T>::sizeInBytes(ptr_a.dims));
        PtrImpl<D,T>::h_malloc_init(ptr_a.dims);
    }

    __host__ inline void allocHostPinned() {
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width; //Here we don't support padding
        gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D,T>::sizeInBytes(ptr_a.dims)));
        PtrImpl<D,T>::h_malloc_init(ptr_a.dims);
    }

    __host__ inline void freePrt() {
        if (ref) {
            ref->cnt--;
            if (ref->cnt == 0) {
                switch (type) {
                    case Device:
                        gpuErrchk(cudaFree(ref->ptr));
                        break;
                    case Host:
                        free(ref->ptr);
                        break;
                    case HostPinned:
                        gpuErrchk(cudaFreeHost(ref->ptr));
                        break;
                    default:
                        break;
                }
                free(ref);
            }
        }
    }

public:

    __host__ inline Ptr() {}

    __host__ inline Ptr(const Ptr<D,T, At>& other) {
        ptr_a = other.ptr_a;
        adjusted_blockSize = other.adjusted_blockSize;
        type = other.type;
        deviceID = other.deviceID;
        if (other.ref) {
            ref = other.ref;
            ref->cnt++;
        }
    }

    __host__ inline Ptr(const PtrDims<D>& dims, MemType type_ = Device, int deviceID_ = 0) {
        allocPtr(dims, type_, deviceID_);
    }

    __host__ inline Ptr(T * data_, const PtrDims<D>& dims, MemType type_ = Device, int deviceID_ = 0) {
        ptr_a.data = data_;
        ptr_a.dims = dims;
        type = type_;
        deviceID = deviceID_;
        adjusted_blockSize = PtrImpl<D,T>::getBlockSize(ptr_a.dims);
    }

    __host__ inline void allocPtr(const PtrDims<D>& dims_, const MemType type_ = Device, const int deviceID_ = 0) {
        ptr_a.dims = dims_;
        type = type_;
        deviceID = deviceID_;
        ref = (RefPtr*)malloc(sizeof(RefPtr));
        ref->cnt = 1;

        switch (type_) {
            case Device:
                allocDevice();
                adjusted_blockSize = PtrImpl<D,T>::getBlockSize(ptr_a.dims);
                break;
            case Host:
                allocHost();
                break;
            case HostPinned:
                allocHostPinned();
                break;
            default:
                break;
        }

        ref->ptr = ptr_a.data;
    }

    __host__ inline ~Ptr() {
        freePrt();
    }

    __host__ inline RawPtr<D,T> d_ptr() const { return ptr_a; }

    __host__ inline operator RawPtr<D,T>() const { return ptr_a; }

    __host__ inline Ptr<D,T, At> crop(Point p, PtrDims<D> newDims) {
        T* ptr = At::point(p, ptr_a.data, ptr_a.dims);
        ref->cnt++;
        const RawPtr<D,T> newRawPtr = {ptr, newDims};
        return {newRawPtr, ref, PtrImpl<D,T>::getBlockSize(newDims), type, deviceID};
    }
    __host__ inline RawPtr<D,T> raw_ptr() const {
        return ptr_a;
    }
    __host__ inline PtrDims<D> dims() const {
        return ptr_a.dims;
    }
    __host__ inline T* data() const {
        return ptr_a.data;
    }
    __host__ inline dim3 getBlockSize() const {
        return adjusted_blockSize;
    }
    __host__ inline MemType getMemType() const {
        return type;
    }
    __host__ inline int getDeviceID() const {
        return deviceID;
    }
};

template <typename T>
class Ptr1D : public Ptr<_1D, T, PtrAccessor<_1D>> {
public:
    __host__ inline Ptr1D(uint num_elems, uint size_in_bytes = 0, MemType type_ = Device, int deviceID_ = 0) : 
                          Ptr<_1D, T, PtrAccessor<_1D>>(PtrDims<_1D>(num_elems, size_in_bytes), type_, deviceID_) {}
    __host__ inline Ptr1D(const Ptr<_1D, T, PtrAccessor<_1D>>& other) : Ptr<_1D, T, PtrAccessor<_1D>>(other) {}
    __host__ inline Ptr1D(T * data_, const PtrDims<_1D>& dims_, MemType type_ = Device, int deviceID_ = 0) :
                          Ptr<_1D, T, PtrAccessor<_1D>>(data_, dims_, type_, deviceID_) {}
    __host__ inline Ptr1D<T> crop1D(Point p, PtrDims<_1D> newDims) {
        return Ptr<_1D, T, PtrAccessor<_1D>>::crop(p, newDims);
    }
};

template <typename T>
class Ptr2D : public Ptr<_2D, T, PtrAccessor<_2D>> {
public:
    __host__ inline Ptr2D(uint width_, uint height_, uint pitch_ = 0, MemType type_ = Device, int deviceID_ = 0) : 
    Ptr<_2D, T, PtrAccessor<_2D>>(PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}
    __host__ inline Ptr2D(const Ptr<_2D, T, PtrAccessor<_2D>>& other) : Ptr<_2D, T, PtrAccessor<_2D>>(other) {}
    __host__ inline Ptr2D(T * data_, uint width_, uint height_, uint pitch_, MemType type_ = Device, int deviceID_ = 0) :
                          Ptr<_2D, T, PtrAccessor<_2D>>(data_, PtrDims<_2D>(width_,height_,pitch_), type_, deviceID_) {}
    __host__ inline Ptr2D<T> crop2D(Point p, PtrDims<_2D> newDims) {
        return Ptr<_2D, T, PtrAccessor<_2D>>::crop(p, newDims);
    }
};

// A Ptr3D pointer can have 2D padding on each plane, or not
template <typename T>
class Ptr3D : public Ptr<_3D, T, PtrAccessor<_3D>> {
public:
    __host__ inline Ptr3D(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_ = 0, MemType type_ = Device, int deviceID_ = 0) : 
    Ptr<_3D, T, PtrAccessor<_3D>>(PtrDims<_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_) {}
    __host__ inline Ptr3D(const Ptr<_3D, T, PtrAccessor<_3D>>& other) : Ptr<_3D, T, PtrAccessor<_3D>>(other) {}
    __host__ inline Ptr3D(T * data_, const PtrDims<_3D>& dims_, MemType type_ = Device, int deviceID_ = 0) :
                          Ptr<_3D, T, PtrAccessor<_3D>>(data_, dims_, type_, deviceID_) {}
    __host__ inline Ptr3D<T> crop3D(Point p, PtrDims<_3D> newDims) {
        return Ptr<_3D, T, PtrAccessor<_3D>>::crop(p, newDims);
    }
};

// A Tensor pointer will never have any padding
template <typename T>
class Tensor : public Ptr<_3D, T, PtrAccessor<_3D>> {
public:
    __host__ inline Tensor(uint width_, uint height_, uint planes_, uint color_planes_ = 1, MemType type_ = Device, int deviceID_ = 0) : 
    Ptr<_3D,T, PtrAccessor<_3D>>(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)*width_), type_, deviceID_) {}
};

template <ND D, typename T>
struct perthread_write {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<D,T>& output) {
        *PtrAccessor<D>::point(thread, output.data, output.dims) = input;
    }
};

template <typename T, typename Enabler=void>
struct perthread_packed2plannar_tensor_write;

template <typename T>
struct perthread_packed2plannar_tensor_write<T, typename std::enable_if_t<CN(T) == 2>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const Tensor<typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr.data, ptr.dims);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
    }
};

template <typename T>
struct perthread_packed2plannar_tensor_write<T, typename std::enable_if_t<CN(T) == 3>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const Tensor<typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr.data, ptr.dims);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.z;
    }
};

template <typename T>
struct perthread_packed2plannar_tensor_write<T, typename std::enable_if_t<CN(T) == 4>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const Tensor<typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr.data, ptr.dims);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.z;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.w;
    }
};

template <ND D, typename T, typename Enabler=void>
struct perthread_split_write;

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 2>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<D,decltype(T::x)>& output1,
                             const RawPtr<D,decltype(T::y)>& output2) {
        *PtrAccessor<D>::point(thread, output1.data, output1.dims) = input.x;
        *PtrAccessor<D>::point(thread, output2.data, output2.dims) = input.y;
    }
};

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 3>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, 
                             const RawPtr<D,decltype(T::x)>& output1, 
                             const RawPtr<D,decltype(T::y)>& output2,
                             const RawPtr<D,decltype(T::z)>& output3) {
        *PtrAccessor<D>::point(thread, output1.data, output1.dims) = input.x;
        *PtrAccessor<D>::point(thread, output2.data, output2.dims) = input.y;
        *PtrAccessor<D>::point(thread, output3.data, output3.dims) = input.z;
    }
};

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 4>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, 
                               RawPtr<D,decltype(T::x)> output1, 
                               RawPtr<D,decltype(T::y)> output2,
                               RawPtr<D,decltype(T::z)> output3,
                               RawPtr<D,decltype(T::w)> output4) { 
        *PtrAccessor<D>::point(thread, output1.data, output1.dims) = input.x;
        *PtrAccessor<D>::point(thread, output2.data, output2.dims) = input.y;
        *PtrAccessor<D>::point(thread, output3.data, output3.dims) = input.z;
        *PtrAccessor<D>::point(thread, output4.data, output4.dims) = input.w;
    }
};

// The following code is a modification of the OpenCV file resize.cu
// which has the following license

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//! interpolation algorithm
enum InterpolationType {
    /** nearest neighbor interpolation */
    INTER_NEAREST        = 0,
    /** bilinear interpolation */
    INTER_LINEAR         = 1,
    /** bicubic interpolation */
    INTER_CUBIC          = 2,
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    INTER_AREA           = 3,
    /** Lanczos interpolation over 8x8 neighborhood */
    INTER_LANCZOS4       = 4,
    /** Bit exact bilinear interpolation */
    INTER_LINEAR_EXACT = 5,
    /** Bit exact nearest neighbor interpolation. This will produce same results as
    the nearest neighbor method in PIL, scikit-image or Matlab. */
    INTER_NEAREST_EXACT  = 6,
    /** mask for interpolation codes */
    INTER_MAX            = 7,
    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    source image, they are set to zero */
    WARP_FILL_OUTLIERS   = 8,
    /** flag, inverse transformation

    For example, #linearPolar or #logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    WARP_INVERSE_MAP     = 16,
    NONE = 17
};

template <ND D, typename I, InterpolationType INTER_T>
struct interpolate_read;

template <typename I>
struct interpolate_read<_2D, I, InterpolationType::INTER_LINEAR> {
    static __device__ __forceinline__ I exec(const uint& x, const uint& y, const I*__restrict__ i_data,
                                                   const uint& width, const uint& height, const uint& pitch,
                                                   const float& fx, const float& fy) {
        const float src_x = x * fx;
        const float src_y = y * fy;

        const uint x1 = __float2int_rd(src_x);
        const uint y1 = __float2int_rd(src_y);
        const uint x2 = x1 + 1;
        const uint y2 = y1 + 1;        
        const uint x2_read = ::min(x2, width - 1);
        const uint y2_read = ::min(y2, height - 1);

        using floatcn_t = typename VectorType<float, VectorTraits<I>::cn>::type;
        floatcn_t out = make_set<floatcn_t>(0.f);
        uchar3 src_reg = *PtrAccessor<_2D>::cr_point(x1, y1, i_data, pitch);  //input.at_cr({x1, y1});
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y1, i_data, pitch); //*input.at_cr({x2_read, y1});
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x1, y2_read, i_data, pitch); //*input.at_cr({x1, y2_read});
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y2_read, i_data, pitch); //*input.at_cr({x2_read, y2_read});
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        return saturate_cast<I>(out);   
    } 
};

template <typename I>
struct interpolate_read<_3D, I, InterpolationType::INTER_LINEAR> {
    FK_DEVICE_FUSE I exec(const Point& thread, const RawPtr<_3D, I>& input) {
        const RawPtr<_2D,I>& myPtr = input.ptrs[thread.z];                                           
        return interpolate_read<_2D, I, InterpolationType::INTER_LINEAR>
                                ::exec(thread, myPtr.data, myPtr.dims, myPtr.fy, myPtr.fx);
    }
};

}
