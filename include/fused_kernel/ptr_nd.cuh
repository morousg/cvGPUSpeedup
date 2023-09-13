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

#include <array>

#include "cuda_vector_utils.cuh"
#include "parameter_pack_utils.cuh"

namespace fk {

struct Size {
    int width;
    int height;
};

struct Point {
    uint x;
    uint y;
    uint z;
    FK_HOST_DEVICE_CNST Point(const uint x_ = 0, const uint y_ = 0, const uint z_ = 0) : x(x_), y(y_), z(z_) {}
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


inline dim3 getBlockSize(const uint& width, const uint& height) {
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
enum ND { _1D=1, _2D=2, _3D=3, T3D=4 };

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
    int width;
    int height;
    int pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0) {}
    __host__ __device__ __forceinline__
    PtrDims(uint width_, uint height_, uint pitch_=0) :
                    width(width_), height(height_), pitch(pitch_) {}
};

template <>
struct PtrDims<_3D> {
    // Image batch shape
    // R,G,B
    // R,G,B
    // R,G,B

    // Width and Height of one individual image
    uint width;
    uint height;
    // Number of images
    uint planes;
    // Number of color channels
    uint color_planes;

    // Pitch for each image
    uint pitch;

    // Pitch to jump one plane
    uint plane_pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0), plane_pitch(0) {}
    __host__ __device__ __forceinline__ 
    PtrDims(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_=0) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(pitch_), plane_pitch(pitch_ * height_) {}
};

template <>
struct PtrDims<T3D> {
    // Image batch shape
    // R,R,R
    // G,G,G
    // B,B,B

    // Width and Height of one individual image
    uint width;
    uint height;
    // Number of images
    uint planes;
    // Number of color channels
    uint color_planes;

    // Pitch for each image
    uint pitch;

    // Pitch to jump one plane
    uint plane_pitch;

    // Pitch to jump to the next plane of the same image
    uint color_planes_pitch;

    __host__ __device__ __forceinline__ PtrDims() : pitch(0), plane_pitch(0), color_planes_pitch(0) {}
    __host__ __device__ __forceinline__
        PtrDims(uint width_, uint height_, uint planes_, uint color_planes_ = 1) :
        width(width_), height(height_), planes(planes_), color_planes(color_planes_),
        pitch(0), plane_pitch(0), color_planes_pitch(0) {}
};

template <ND D, typename T>
struct RawPtr;

template <typename T>
struct RawPtr<_1D, T> {
    T* data;
    PtrDims<_1D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=cn<T>};
};

template <typename T>
struct RawPtr<_2D, T> {
    T* data;
    PtrDims<_2D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=cn<T>};
};

template <typename T>
struct RawPtr<_3D, T> {
    T* data;
    PtrDims<_3D> dims;
    using base = typename VectorTraits<T>::base;
    enum {cn=cn<T>};
};

template <typename T>
struct RawPtr<T3D, T> {
    T* data;
    PtrDims<T3D> dims;
    using base = typename VectorTraits<T>::base;
    enum { cn = cn<T> };
};

template <ND D>
struct PtrAccessor;

template <>
struct PtrAccessor<_1D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE const T*__restrict__ cr_point(const Point& p, const RawPtr<_1D, T>& ptr) {
        return (const T*)(ptr.data + p.x);
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const RawPtr<_1D, T>& ptr) {
        return ptr.data + p.x;
    }
};

template <>
struct PtrAccessor<_2D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE const T*__restrict__ cr_point(const Point& p, const RawPtr<_2D, T>& ptr) {
        return (const T*)((const char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const RawPtr<_2D, T>& ptr) {
        return (T*)((char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
    }
};

template <>
struct PtrAccessor<_3D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE const T*__restrict__ cr_point(const Point& p, const RawPtr<_3D, T>& ptr) {
        return (const T*)((const char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const RawPtr<_3D, T>& ptr) {
        return (T*)((char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
    }
};

template <>
struct PtrAccessor<T3D> {
    template <typename T>
    FK_HOST_DEVICE_FUSE const T* __restrict__ cr_point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
        return (const T*)((const char*)ptr.data + (color_plane * ptr.dims.color_plane_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
    }

    template <typename T>
    static __device__ __forceinline__ __host__ T* point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
        return (T*)((char*)ptr.data + (color_plane * ptr.dims.color_plane_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
    }
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
            ptr_a.dims.pitch = (int)pitch;
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
        return dims.pitch * dims.height * dims.planes * dims.color_planes;
    }
    FK_HOST_FUSE uint getNumElements(const PtrDims<_3D>& dims) {
        return dims.width * dims.height * dims.planes * dims.color_planes;
    }
    FK_HOST_FUSE void d_malloc(RawPtr<_3D,T>& ptr_a) {
        if (ptr_a.dims.pitch == 0) {
            throw std::exception(); // Not supported to have 2D pitch in a 3D pointer
        } else {
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_3D,T>::sizeInBytes(ptr_a.dims)));
        }
        ptr_a.dims.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.height;
    }
    FK_HOST_FUSE void h_malloc_init(PtrDims<_3D>& dims) {
        dims.plane_pitch = dims.pitch * dims.height;
    }
    FK_HOST_FUSE dim3 getBlockSize(const PtrDims<_3D>& dims) {
        return fk::getBlockSize(dims.width, dims.height);
    }
};

template <typename T>
struct PtrImpl<T3D, T> {
    FK_HOST_FUSE size_t sizeInBytes(const PtrDims<T3D>& dims) {
        return dims.color_planes_pitch * dims.color_planes;
    }
    FK_HOST_FUSE uint getNumElements(const PtrDims<T3D>& dims) {
        return dims.width * dims.height * dims.planes * dims.color_planes;
    }
    FK_HOST_FUSE void d_malloc(RawPtr<T3D, T>& ptr_a) {
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
        ptr_a.dims.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.height;
        ptr_a.dims.color_planes_pitch = ptr_a.dims.plane_pitch * ptr_a.dims.color_planes;
        gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<T3D, T>::sizeInBytes(ptr_a.dims)));
    }
    FK_HOST_FUSE void h_malloc_init(PtrDims<T3D>& dims) {
        dims.pitch = sizeof(T) * dims.width;
        dims.plane_pitch = dims.pitch * dims.height;
        dims.color_planes_pitch = dims.plane_pitch * dims.color_planes;
    }
    FK_HOST_FUSE dim3 getBlockSize(const PtrDims<T3D>& dims) {
        return fk::getBlockSize(dims.width, dims.height);
    }
};

template <ND D, typename T>
class Ptr{

using At = PtrAccessor<D>;

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

    __host__ inline constexpr Ptr(const RawPtr<D,T>& ptr_a_, RefPtr* ref_, const dim3& bs_, const MemType& type_, const int& devID) : 
                                  ptr_a(ptr_a_), ref(ref_), adjusted_blockSize(bs_), type(type_), deviceID(devID) {}
    
    __host__ inline constexpr void allocDevice() {
        int currentDevice;
        gpuErrchk(cudaGetDevice(&currentDevice));
        gpuErrchk(cudaSetDevice(deviceID));
        PtrImpl<D,T>::d_malloc(ptr_a);
        if (currentDevice != deviceID) {
            gpuErrchk(cudaSetDevice(currentDevice));
        }
    }

    __host__ inline constexpr void allocHost() {
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width; //Here we don't support padding
        ptr_a.data = (T*)malloc(PtrImpl<D,T>::sizeInBytes(ptr_a.dims));
        PtrImpl<D,T>::h_malloc_init(ptr_a.dims);
    }

    __host__ inline constexpr void allocHostPinned() {
        ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width; //Here we don't support padding
        gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D,T>::sizeInBytes(ptr_a.dims)));
        PtrImpl<D,T>::h_malloc_init(ptr_a.dims);
    }

    __host__ inline constexpr void freePrt() {
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

    __host__ inline constexpr Ptr() {}

    __host__ inline constexpr Ptr(const Ptr<D,T>& other) {
        ptr_a = other.ptr_a;
        adjusted_blockSize = other.adjusted_blockSize;
        type = other.type;
        deviceID = other.deviceID;
        if (other.ref) {
            ref = other.ref;
            ref->cnt++;
        }
    }

    __host__ inline constexpr Ptr(const PtrDims<D>& dims, const MemType& type_ = Device, const int& deviceID_ = 0) {
        allocPtr(dims, type_, deviceID_);
    }

    __host__ inline constexpr Ptr(T * data_, const PtrDims<D>& dims, const MemType& type_ = Device, const int& deviceID_ = 0) {
        ptr_a.data = data_;
        ptr_a.dims = dims;
        type = type_;
        deviceID = deviceID_;
        adjusted_blockSize = PtrImpl<D,T>::getBlockSize(ptr_a.dims);
    }

    __host__ inline constexpr void allocPtr(const PtrDims<D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) {
        ptr_a.dims = dims_;
        type = type_;
        deviceID = deviceID_;
        ref = (RefPtr*)malloc(sizeof(RefPtr));
        ref->cnt = 1;

        switch (type) {
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

    __host__ inline constexpr RawPtr<D,T> ptr() const { return ptr_a; }

    __host__ inline constexpr operator RawPtr<D,T>() const { return ptr_a; }

    __host__ inline constexpr Ptr<D,T> crop(const Point& p, const PtrDims<D>& newDims) {
        T* ptr = At::point(p, ptr_a);
        ref->cnt++;
        const RawPtr<D,T> newRawPtr = {ptr, newDims};
        return {newRawPtr, ref, PtrImpl<D,T>::getBlockSize(newDims), type, deviceID};
    }
    __host__ inline constexpr PtrDims<D> dims() const {
        return ptr_a.dims;
    }
    __host__ inline dim3 getBlockSize() const {
        return adjusted_blockSize;
    }
    __host__ inline constexpr MemType getMemType() const {
        return type;
    }
    __host__ inline constexpr int getDeviceID() const {
        return deviceID;
    }

    __host__ inline constexpr size_t sizeInBytes() const {
        return PtrImpl<D, T>::sizeInBytes(ptr_a.dims);
    }

    __host__ inline constexpr uint getNumElements() const {
        return PtrImpl<D, T>::getNumElements(ptr_a.dims);
    }

    __host__ inline constexpr void setTo(const T& val) {
        if (type == MemType::Host || type == MemType::HostPinned) {
            for (int i = 0; i < (int)getNumElements(); i++) {
                ptr_a.data[i] = val;
            }
        } else {
            throw std::exception();
        }
        
    }
};

template <typename T>
class Ptr1D : public Ptr<_1D, T> {
public:
    __host__ inline constexpr Ptr1D<T>(const uint& num_elems, const uint& size_in_bytes = 0, const MemType& type_ = Device, const int& deviceID_ = 0) : 
                                    Ptr<_1D, T>(PtrDims<_1D>(num_elems, size_in_bytes), type_, deviceID_) {}

    __host__ inline constexpr Ptr1D<T>(const Ptr<_1D, T>& other) : Ptr<_1D, T>(other) {}

    __host__ inline constexpr Ptr1D<T>(T * data_, const PtrDims<_1D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
                          Ptr<_1D, T>(data_, dims_, type_, deviceID_) {}

    __host__ inline constexpr Ptr1D<T> crop1D(const Point& p, const PtrDims<_1D>& newDims) { return Ptr<_1D, T>::crop(p, newDims); }
};

template <typename T>
class Ptr2D : public Ptr<_2D, T> {
public:
    __host__ inline constexpr Ptr2D<T>() {}
    __host__ inline constexpr Ptr2D<T>(const uint& width_, const uint& height_, const uint& pitch_ = 0, const MemType& type_ = Device, const int& deviceID_ = 0) :
                                    Ptr<_2D, T>(PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}

    __host__ inline constexpr Ptr2D<T>(const Ptr<_2D, T>& other) : Ptr<_2D, T>(other) {}

    __host__ inline constexpr Ptr2D<T>(T * data_, const uint& width_, const uint& height_, const uint& pitch_, const MemType& type_ = Device, const int& deviceID_ = 0) :
                                    Ptr<_2D, T>(data_, PtrDims<_2D>(width_,height_,pitch_), type_, deviceID_) {}

    __host__ inline constexpr Ptr2D<T> crop2D(const Point& p, const PtrDims<_2D>& newDims) { return Ptr<_2D, T>::crop(p, newDims); }
};

// A Ptr3D pointer
template <typename T>
class Ptr3D : public Ptr<_3D, T> {
public:
    __host__ inline constexpr Ptr3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const uint& pitch_ = 0, const MemType& type_ = Device, const int& deviceID_ = 0) : 
                          Ptr<_3D, T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_) {}

    __host__ inline constexpr Ptr3D<T>(const Ptr<_3D, T>& other) : Ptr<_3D, T>(other) {}
    
    __host__ inline constexpr Ptr3D<T>(T * data_, const PtrDims<_3D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
                          Ptr<_3D, T>(data_, dims_, type_, deviceID_) {}

    __host__ inline constexpr Ptr3D<T> crop3D(const Point& p, const PtrDims<_3D>& newDims) { return Ptr<_3D, T>::crop(p, newDims); }
};

// A color plane transposed 3D pointer PtrT3D
template <typename T>
class PtrT3D : public Ptr<T3D, T> {
public:
    __host__ inline constexpr PtrT3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
        Ptr<T3D, T>(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

    __host__ inline constexpr PtrT3D<T>(const Ptr<T3D, T>& other) : Ptr<T3D, T>(other) {}

    __host__ inline constexpr PtrT3D<T>(T* data_, const PtrDims<T3D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
        Ptr<T3D, T>(data_, dims_, type_, deviceID_) {}

    __host__ inline constexpr PtrT3D<T> crop3D(const Point& p, const PtrDims<T3D>& newDims) { return Ptr<T3D, T>::crop(p, newDims); }
};

// A Tensor pointer will never have any padding
template <typename T>
class Tensor : public Ptr<_3D, T> {
public:
    __host__ inline constexpr Tensor() {}

    __host__ inline constexpr Tensor(const Tensor<T>& other) : Ptr<_3D, T>(other) {}
    
    __host__ inline constexpr Tensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) : 
                                     Ptr<_3D,T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)*width_), type_, deviceID_) {}
    
    __host__ inline constexpr Tensor(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) : 
                                     Ptr<_3D,T>(data, PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)*width_), type_, deviceID_) {}
    
    __host__ inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) {
                                               this->allocPtr(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)*width_), type_, deviceID_);
    }
};
}
