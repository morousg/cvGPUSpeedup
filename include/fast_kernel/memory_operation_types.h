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
#include "cuda_vector_utils.h"
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

template <typename T>
struct Device_Ptr_2D {
    T* data;
    uint width;
    uint height;
    uint pitch;

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
class Ptr_2D {

private:
    struct refPtr {
        void* ptr;
        int cnt;  
    };
    refPtr* ref{ nullptr };
    Device_Ptr_2D<T> device_ptr;

    __host__ error_t free_ptr() {
        cudaError_t err = cudaSuccess;
        if (ref) {
            ref->cnt--;
            if (ref->cnt == 0) {
                err = cudaFree(ref->ptr);
                free(ref);
            }
        }
        return err;
    }

    __host__ Ptr_2D(T * data_, refPtr * ref_, uint width_, uint height_, uint pitch_) : ref(ref_) {
        device_ptr.data = data_;
        device_ptr.width = width_;
        device_ptr.height = height_;
        device_ptr.pitch = pitch_;
    }

public:

    __host__ Ptr_2D() {}
    __host__ Ptr_2D(const Ptr_2D<T>& other) {
        device_ptr = other.device_ptr;
        if (other.ref) {
            ref = other.ref;
            ref->cnt++;
        }
    }
    __host__ Ptr_2D(uint width_, uint height_, uint pitch_ = 0) {
        alloc_ptr(width_, height_, pitch_);
    }
    __host__ Ptr_2D(T * data_, uint width_, uint height_, uint pitch_) {
        device_ptr.data = data_;
        device_ptr.width = width_;
        device_ptr.height = height_;
        device_ptr.pitch = pitch_;
    }
    __host__ ~Ptr_2D() {
        // TODO: add gpuCkeck
        free_ptr();
    }

    __host__ Device_Ptr_2D<T> d_ptr() const { return device_ptr; }

    operator Device_Ptr_2D<T>() const { return device_ptr; }

    __host__ error_t alloc_ptr(uint width_, uint height_, uint pitch_ = 0) {
        device_ptr.width = width_;
        device_ptr.height = height_;
        ref = (refPtr*)malloc(sizeof(refPtr));
        ref->cnt = 1;
        error_t err;
        if (pitch_ == 0) {
            size_t pitch_temp;
            err = cudaMallocPitch(&device_ptr.data, &pitch_temp, sizeof(T) * device_ptr.width, device_ptr.height);
            device_ptr.pitch = pitch_temp;
        } else {
            device_ptr.pitch = pitch_;
            err = cudaMalloc(&device_ptr.data, device_ptr.pitch * device_ptr.height);
        }
        ref->ptr = device_ptr.data;

        return err;
    }

    __host__ Ptr_2D<T> crop(Point p, uint width_n, uint height_n) {
        T* ptr = device_ptr.at(p);
        ref->cnt++;
        return {ptr, ref, width_n, height_n, device_ptr.pitch};
    }

    __host__ void setResize(Resize& resize_) {
        device_ptr.resize = resize_;
    }

    __host__ uint width() const {
        return device_ptr.width;
    }
    __host__ uint height() const {
        return device_ptr.height;
    }
    __host__ uint pitch() const {
        return device_ptr.pitch;
    }
    __host__ T* data() const {
        return device_ptr.data;
    }

    __host__ dim3 getBlockSize() const {
        const std::unordered_map<uint, uint> optionsYX = {{8, 32}, {7, 32}, {6, 32}, {5, 32}, {4, 64}, {3, 64}, {2, 128}, {1, 256}};
        const std::unordered_map<uint, uint> scoresY = {{8, 4}, {7, 3}, {6, 2}, {5, 1}, {4, 4}, {3, 3}, {2, 4}, {1, 4}};

        std::vector<uint> zeroModY;

        for (uint i = min(8,device_ptr.height); i > 0; i--) {
            if (device_ptr.height % i == 0) {
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
    __device__ void operator()(I input, Device_Ptr_2D<O> output) {
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
                               Device_Ptr_2D<decltype(I::x)> output1,
                               Device_Ptr_2D<decltype(I::y)> output2) {
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
                               Device_Ptr_2D<decltype(I::x)> output1, 
                               Device_Ptr_2D<decltype(I::y)> output2,
                               Device_Ptr_2D<decltype(I::z)> output3) {
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
                               Device_Ptr_2D<decltype(I::x)> output1, 
                               Device_Ptr_2D<decltype(I::y)> output2,
                               Device_Ptr_2D<decltype(I::z)> output3,
                               Device_Ptr_2D<decltype(I::w)> output4) { 
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