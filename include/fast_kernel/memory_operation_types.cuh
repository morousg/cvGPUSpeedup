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
#include "cuda_vector_utils.cuh"
#include <cooperative_groups.h>
#include <vector>
#include <unordered_map>

namespace cg = cooperative_groups;

namespace fk {

enum InterpType {INTER_NEAREST , INTER_LINEAR, INTER_CUBIC, NONE};
enum BorderMode {BORDER_REFLECT101 , BORDER_REPLICATE , BORDER_CONSTANT , BORDER_REFLECT, BORDER_WRAP};

struct Point {
  uint x;
  uint y;  
};

struct Crop {
    Point start_p;
    uint width_c;
    uint height_c;
};

struct Resize {
    InterpType inter_t;
    BorderMode border_m;
    uint target_width;
    uint target_height;
};

enum MemType { Device, Host, HostPinned };

template <typename T>
struct MemPatterns {
    T* data;
    uint width;
    uint height;
    uint planes;
    uint pitch;
    MemType type;
    int deviceID;

    Resize resize{ InterpType::NONE, BorderMode::BORDER_CONSTANT, 0, 0 };

    template <typename C>
    __host__ __device__
    constexpr inline C* getPtrUnsafe(Point p) const {
        if (p.x >= width || p.y >= height) {
            return nullptr;
        } else if (hasPadding()) {
            return (C*)((uchar*)data + (p.y * pitch)) + p.x;
        }
        return (C*)(data) + (p.y * width) + p.x;
    }

    template <typename C>
    __host__ __device__ 
    constexpr inline bool castIsOutOfBounds() const {
        return sizeInBytes() % sizeof(C) == 0;
    }

    __host__ __device__ 
    constexpr inline uint sizeInBytes() const {
        return pitch * height;
    }

    __host__ __device__
    constexpr inline bool hasPadding() const {
        return pitch != sizeof(T) * width;
    }

    __host__ __device__ 
    constexpr inline T* at(Point p) const {
        return getPtrUnsafe<T>(p);
    }

    template <typename C>
    __host__ __device__ 
    constexpr inline C* getCastPtrFor(Point p) const {
        if (hasPadding() || castIsOutOfBounds()) {
            // Not supported by now
            return nullptr;
        }
        return getPtrUnsafe<C>(p);
    }
    __host__ __device__ 
    constexpr inline T getResizedPixel(Point p) const {
        // Do all the magic
        return (T)0;
    }

    __host__ __device__ 
    constexpr inline uint getNumElements() const {
        return width * height;
    }
};

template <typename T>
class Ptr3D {

private:
    struct refPtr {
        void* ptr;
        int cnt;  
    };
    refPtr* ref{ nullptr };
    MemPatterns<T> patterns;

    __host__ void freePrt() {
        if (ref) {
            ref->cnt--;
            if (ref->cnt == 0) {
                switch (patterns.type) {
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

    __host__ Ptr3D(T * data_, refPtr * ref_, uint width_, uint height_, uint pitch_, uint planes_, MemType type_, int deviceID_) : ref(ref_) {
        patterns.data = data_;
        patterns.width = width_;
        patterns.height = height_;
        patterns.pitch = pitch_;
        patterns.planes = planes_;
        patterns.type = type_;
        patterns.deviceID = deviceID_;
    }

    __host__ void allocDevice() {
        int currentDevice;
        gpuErrchk(cudaGetDevice(&currentDevice));
        gpuErrchk(cudaSetDevice(patterns.deviceID));
        if (patterns.pitch == 0) {
            size_t pitch_temp;
            gpuErrchk(cudaMallocPitch(&patterns.data, &pitch_temp, sizeof(T) * patterns.width, patterns.height * patterns.planes));
            patterns.pitch = pitch_temp;
        } else {
            gpuErrchk(cudaMalloc(&patterns.data, patterns.pitch * patterns.height * patterns.planes));
        }
        if (currentDevice != patterns.deviceID) {
            gpuErrchk(cudaSetDevice(currentDevice));
        }
    }

    __host__ void allocHost() {
        patterns.data = (T*)malloc(sizeof(T) * patterns.width * patterns.height * patterns.planes);
        patterns.pitch = sizeof(T) * patterns.width; //Here we don't support padding
    }

    __host__ void allocHostPinned() {
        gpuErrchk(cudaMallocHost(&patterns.data, sizeof(T) * patterns.width * patterns.height * patterns.planes));
        patterns.pitch = sizeof(T) * patterns.width; //Here we don't support padding
    }

public:

    __host__ Ptr3D() {}

    __host__ Ptr3D(const Ptr3D<T>& other) {
        patterns = other.patterns;
        if (other.ref) {
            ref = other.ref;
            ref->cnt++;
        }
    }

    __host__ Ptr3D(uint width_, uint height_, uint pitch_ = 0, uint planes_ = 1, MemType type_ = Device, int deviceID_ = 0) {
        allocPtr(width_, height_, pitch_, planes_, type_, deviceID_);
    }

    __host__ Ptr3D(T * data_, uint width_, uint height_, uint pitch_, uint planes_ = 1, MemType type_ = Device, int deviceID_ = 0) {
        patterns.data = data_;
        patterns.width = width_;
        patterns.height = height_;
        patterns.pitch = pitch_;
        patterns.planes = planes_;
        patterns.type = type_;
        patterns.deviceID = deviceID_;
    }

    __host__ ~Ptr3D() {
        // TODO: add gpuCkeck
        freePrt();
    }

    __host__ MemPatterns<T> d_ptr() const { return patterns; }

    operator MemPatterns<T>() const { return patterns; }

    __host__ void allocPtr(uint width_, uint height_, uint pitch_ = 0, uint planes_ = 1, MemType type_ = Device, int deviceID_ = 0) {
        patterns.width = width_;
        patterns.height = height_;
        patterns.pitch = pitch_;
        patterns.planes = planes_;
        patterns.type = type_;
        patterns.deviceID = deviceID_;
        ref = (refPtr*)malloc(sizeof(refPtr));
        ref->cnt = 1;

        switch (type_) {
            case Device:
                allocDevice();
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

        ref->ptr = patterns.data;
    }

    __host__ Ptr3D<T> crop(Point p, uint width_n, uint height_n, uint planes_n = 1) {
        T* ptr = patterns.at(p);
        ref->cnt++;
        return {ptr, ref, width_n, height_n, patterns.pitch, planes_n, patterns.type, patterns.deviceID};
    }

    __host__ void setResize(Resize& resize_) {
        patterns.resize = resize_;
    }

    __host__ uint width() const {
        return patterns.width;
    }
    __host__ uint height() const {
        return patterns.height;
    }
    __host__ uint pitch() const {
        return patterns.pitch;
    }
    __host__ T* data() const {
        return patterns.data;
    }

    __host__ dim3 getBlockSize() const {
        const std::unordered_map<uint, uint> optionsYX = {{8, 32}, {7, 32}, {6, 32}, {5, 32}, {4, 64}, {3, 64}, {2, 128}, {1, 256}};
        const std::unordered_map<uint, uint> scoresY = {{8, 4}, {7, 3}, {6, 2}, {5, 1}, {4, 4}, {3, 3}, {2, 4}, {1, 4}};

        std::vector<uint> zeroModY;

        for (uint i = min(8,patterns.height); i > 0; i--) {
            if (patterns.height % i == 0) {
                zeroModY.push_back(i);
            }
        }

        uint currentScore = 0;
        uint currentY = 1;
        for (auto& ySize : zeroModY) {
            if (scoresY.at(ySize) > currentScore) {
                currentScore = scoresY.at(ySize);
                currentY = ySize;
            }
            if (currentScore == 4) {
                break;
            }
        }

        return dim3(optionsYX.at(currentY), currentY);
    }
};

template <typename I, typename O=I>
struct perthread_write_2D {
    __device__ void operator()(I input, MemPatterns<O> output) {
        cg::thread_block g =  cg::this_thread_block();
        uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
        uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;

        *(output.at({x, y})) = input;
    }
};

template <typename I, typename Enabler=void>
struct perthread_split_write_2D;

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 2>> {
    __device__ void operator()(I input,
                               MemPatterns<decltype(I::x)> output1,
                               MemPatterns<decltype(I::y)> output2) {
        cg::thread_block g =  cg::this_thread_block();
        Point p;
        p.x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
        p.y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y;
    }
};

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 3>> {
    __device__ void operator()(I input, 
                               MemPatterns<decltype(I::x)> output1, 
                               MemPatterns<decltype(I::y)> output2,
                               MemPatterns<decltype(I::z)> output3) {
        cg::thread_block g =  cg::this_thread_block();
        Point p;
        p.x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
        p.y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y; 
        *output3.at(p) = input.z;
    }
};

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<NUM_COMPONENTS(I) == 4>> {
    __device__ void operator()(I input, 
                               MemPatterns<decltype(I::x)> output1, 
                               MemPatterns<decltype(I::y)> output2,
                               MemPatterns<decltype(I::z)> output3,
                               MemPatterns<decltype(I::w)> output4) { 
        cg::thread_block g =  cg::this_thread_block();
        Point p;
        p.x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
        p.y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y; 
        *output3.at(p) = input.z;
        *output4.at(p) = input.w; 
    }
};

}