/* Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_PTR_ND_CUH
#define FK_PTR_ND_CUH

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/ptr_nd.h>

namespace fk {

    template <ND D>
    struct PtrAccessor;

    template <>
    struct PtrAccessor<_1D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* __restrict__ cr_point(const Point& p, const RawPtr<_1D, T>& ptr) {
            return ((const BiggerType*)ptr.data) + p.x;
        }

        template <typename T, typename BiggerType = T>
        static __device__ __forceinline__ __host__ BiggerType* point(const Point& p, const RawPtr<_1D, T>& ptr) {
            return (BiggerType*)ptr.data + p.x;
        }
    };

    template <>
    struct PtrAccessor<_2D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* __restrict__ cr_point(const Point& p, const RawPtr<_2D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        static __device__ __forceinline__ __host__ BiggerType* point(const Point& p, const RawPtr<_2D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<_3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* __restrict__ cr_point(const Point& p, const RawPtr<_3D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        static __device__ __forceinline__ __host__ BiggerType* point(const Point& p, const RawPtr<_3D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<T3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* __restrict__ cr_point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
            return (const BiggerType*)((const char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        static __device__ __forceinline__ __host__ BiggerType* point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
            return (BiggerType*)((char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }
    };

    template<ND D>
    struct StaticPtrAccessor;

    template<>
    struct StaticPtrAccessor<_1D> {
        template <int W, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims1D<W>, T>& ptr) {
            return ptr.data[p.x];
        }

        template <int W, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims1D<W>, T>& ptr, const T& value) {
            ptr.data[p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<_2D> {
        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr) {
            return ptr.data[p.y][p.x];
        }

        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr, const T& value) {
            ptr.data[p.y][p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<_3D> {
        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr) {
            return ptr.data[p.z][p.y][p.x];
        }

        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr, const T& value) {
            ptr.data[p.z][p.y][p.x] = value;
        }
    };

    template <typename StaticRawPtr>
    struct StaticPtr {
        using Type = typename StaticRawPtr::type;
        using At = StaticPtrAccessor<StaticRawPtr::ND>;
        StaticRawPtr ptr_a;
        inline constexpr StaticRawPtr ptr() const {
            return ptr_a;
        }
        inline constexpr auto dims() const {
            return ptr_a.dims;
        }
    };

    template <ND D, typename T>
    struct PtrImpl;

    template <typename T>
    struct PtrImpl<_1D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_1D>& dims) {
            return dims.pitch;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_1D>& dims) {
            return dims.width;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_1D, T>& ptr_a) {
            gpuErrchk(cudaMalloc(&ptr_a.data, sizeof(T) * ptr_a.dims.width));
            ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_1D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
        }
        FK_HOST_STATIC dim3 getBlockSize(const PtrDims<_1D>& dims) {
            return getDefaultBlockSize(dims.width, 1);
        }
    };

    template <typename T>
    struct PtrImpl<_2D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_2D>& dims) {
            return dims.pitch * dims.height;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_2D>& dims) {
            return dims.width * dims.height;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_2D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                size_t pitch;
                gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
                ptr_a.dims.pitch = (int)pitch;
            } else {
                gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_2D, T>::sizeInBytes(ptr_a.dims)));
            }
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_2D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
        }
        FK_HOST_STATIC dim3 getBlockSize(const PtrDims<_2D>& dims) {
            return getDefaultBlockSize(dims.width, dims.height);
        }
    };

    template <typename T>
    struct PtrImpl<_3D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_3D>& dims) {
            return dims.pitch * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_3D>& dims) {
            return dims.width * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_3D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                throw std::exception(); // Not supported to have 2D pitch in a 3D pointer
            } else {
                gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_3D, T>::sizeInBytes(ptr_a.dims)));
            }
            ptr_a.dims.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.height;
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_3D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
            dims.plane_pitch = dims.pitch * dims.height;
        }
        FK_HOST_STATIC dim3 getBlockSize(const PtrDims<_3D>& dims) {
            return getDefaultBlockSize(dims.width, dims.height);
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
            PtrImpl<T3D, T>::h_malloc_init(ptr_a.dims);
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<T3D, T>::sizeInBytes(ptr_a.dims)));
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<T3D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
            dims.plane_pitch = dims.pitch * dims.height;
            dims.color_planes_pitch = dims.plane_pitch * dims.planes;
        }
        FK_HOST_STATIC dim3 getBlockSize(const PtrDims<T3D>& dims) {
            return getDefaultBlockSize(dims.width, dims.height);
        }
    };

    template <ND D, typename T>
    class Ptr {
        using Type = T;
        using At = PtrAccessor<D>;

    protected:
        struct RefPtr {
            void* ptr;
            int cnt;
        };
        RefPtr* ref{ nullptr };
        RawPtr<D, T> ptr_a;
        dim3 adjusted_blockSize;
        MemType type;
        int deviceID;

        inline constexpr Ptr(const RawPtr<D, T>& ptr_a_, RefPtr* ref_, const dim3& bs_, const MemType& type_, const int& devID) :
            ptr_a(ptr_a_), ref(ref_), adjusted_blockSize(bs_), type(type_), deviceID(devID) {}

        inline constexpr void allocDevice() {
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::d_malloc(ptr_a);
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
        }

        inline constexpr void allocHost() {
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            ptr_a.data = (T*)malloc(PtrImpl<D, T>::sizeInBytes(ptr_a.dims));

        }

        inline constexpr void allocHostPinned() {
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D, T>::sizeInBytes(ptr_a.dims)));
        }

        inline constexpr void freePrt() {
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

        inline constexpr void initFromOther(const Ptr<D, T>& other) {
            ptr_a = other.ptr_a;
            adjusted_blockSize = other.adjusted_blockSize;
            type = other.type;
            deviceID = other.deviceID;
            if (other.ref) {
                ref = other.ref;
                ref->cnt++;
            }
        }

    public:

        inline constexpr Ptr() {}

        inline constexpr Ptr(const Ptr<D, T>& other) {
            initFromOther(other);
        }

        inline constexpr Ptr(const PtrDims<D>& dims, const MemType& type_ = Device, const int& deviceID_ = 0) {
            allocPtr(dims, type_, deviceID_);
        }

        inline constexpr Ptr(T* data_, const PtrDims<D>& dims, const MemType& type_ = Device, const int& deviceID_ = 0) {
            ptr_a.data = data_;
            ptr_a.dims = dims;
            type = type_;
            deviceID = deviceID_;
            adjusted_blockSize = PtrImpl<D, T>::getBlockSize(ptr_a.dims);
        }

        inline constexpr void allocPtr(const PtrDims<D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) {
            ptr_a.dims = dims_;
            type = type_;
            deviceID = deviceID_;
            ref = (RefPtr*)malloc(sizeof(RefPtr));
            ref->cnt = 1;

            switch (type) {
            case Device:
                allocDevice();
                adjusted_blockSize = PtrImpl<D, T>::getBlockSize(ptr_a.dims);
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

        inline ~Ptr() {
            freePrt();
        }

        inline constexpr RawPtr<D, T> ptr() const { return ptr_a; }

        inline constexpr operator RawPtr<D, T>() const { return ptr_a; }

        inline constexpr Ptr<D, T> crop(const Point& p, const PtrDims<D>& newDims) {
            T* ptr = At::point(p, ptr_a);
            ref->cnt++;
            const RawPtr<D, T> newRawPtr = { ptr, newDims };
            return { newRawPtr, ref, PtrImpl<D,T>::getBlockSize(newDims), type, deviceID };
        }
        inline constexpr PtrDims<D> dims() const {
            return ptr_a.dims;
        }
        inline dim3 getBlockSize() const {
            return adjusted_blockSize;
        }
        inline constexpr MemType getMemType() const {
            return type;
        }
        inline constexpr int getDeviceID() const {
            return deviceID;
        }

        inline constexpr size_t sizeInBytes() const {
            return PtrImpl<D, T>::sizeInBytes(ptr_a.dims);
        }

        inline constexpr uint getNumElements() const {
            return PtrImpl<D, T>::getNumElements(ptr_a.dims);
        }

        inline constexpr int getRefCount() const {
            return ref->cnt;
        }

        Ptr<D, T>& operator=(const Ptr<D, T>& other) {
            initFromOther(other);
            return *this;
        }
    };

    template <typename T>
    class Ptr1D : public Ptr<_1D, T> {
    public:
        inline constexpr Ptr1D<T>() {}
        inline constexpr Ptr1D<T>(const uint& num_elems, const uint& size_in_bytes = 0, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_1D, T>(PtrDims<_1D>(num_elems, size_in_bytes), type_, deviceID_) {}

        inline constexpr Ptr1D<T>(const Ptr<_1D, T>& other) : Ptr<_1D, T>(other) {}

        inline constexpr Ptr1D<T>(T* data_, const PtrDims<_1D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_1D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr Ptr1D<T> crop1D(const Point& p, const PtrDims<_1D>& newDims) { return Ptr<_1D, T>::crop(p, newDims); }
    };

    template <typename T>
    class Ptr2D : public Ptr<_2D, T> {
    public:
        inline constexpr Ptr2D<T>() {}
        inline constexpr Ptr2D<T>(const uint& width_, const uint& height_, const uint& pitch_ = 0, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_2D, T>(PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr2D<T>(const Ptr<_2D, T>& other) : Ptr<_2D, T>(other) {}

        inline constexpr Ptr2D<T>(T* data_, const uint& width_, const uint& height_, const uint& pitch_, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_2D, T>(data_, PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr2D<T> crop2D(const Point& p, const PtrDims<_2D>& newDims) { return Ptr<_2D, T>::crop(p, newDims); }
    };

    // A Ptr3D pointer
    template <typename T>
    class Ptr3D : public Ptr<_3D, T> {
    public:
        inline constexpr Ptr3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const uint& pitch_ = 0, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_3D, T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr3D<T>(const Ptr<_3D, T>& other) : Ptr<_3D, T>(other) {}

        inline constexpr Ptr3D<T>(T* data_, const PtrDims<_3D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_3D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr Ptr3D<T> crop3D(const Point& p, const PtrDims<_3D>& newDims) { return Ptr<_3D, T>::crop(p, newDims); }
    };

    // A color-plane-transposed 3D pointer PtrT3D
    template <typename T>
    class PtrT3D : public Ptr<T3D, T> {
    public:
        inline constexpr PtrT3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<T3D, T>(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr PtrT3D<T>(const Ptr<T3D, T>& other) : Ptr<T3D, T>(other) {}

        inline constexpr PtrT3D<T>(T* data_, const PtrDims<T3D>& dims_, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<T3D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr PtrT3D<T> crop3D(const Point& p, const PtrDims<T3D>& newDims) { return Ptr<T3D, T>::crop(p, newDims); }
    };

    // A Tensor pointer
    template <typename T>
    class Tensor : public Ptr<_3D, T> {
    public:
        inline constexpr Tensor() {}

        inline constexpr Tensor(const Tensor<T>& other) : Ptr<_3D, T>(other) {}

        inline constexpr Tensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_3D, T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr Tensor(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<_3D, T>(data, PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) {
            this->allocPtr(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T) * width_), type_, deviceID_);
        }
    };

    // A color-plane-transposed Tensor pointer
    template <typename T>
    class TensorT : public Ptr<T3D, T> {
    public:
        inline constexpr TensorT() {}

        inline constexpr TensorT(const TensorT<T>& other) : Ptr<T3D, T>(other) {}

        inline constexpr TensorT(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<T3D, T>(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr TensorT(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) :
            Ptr<T3D, T>(data, PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = Device, const int& deviceID_ = 0) {
            this->allocPtr(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_);
        }
    };

} // namespace fk

#endif
