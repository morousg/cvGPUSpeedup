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

#ifndef FK_PTR_ND
#define FK_PTR_ND

#include <fused_kernel/core/utils/utils.h>

namespace fk {
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
        static constexpr inline void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
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
        static constexpr inline void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 1;
                if constexpr (bxS_t == 3) return;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t + 1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <>
    struct computeBestSolution<3, 1> {
        static constexpr inline void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = 3;
                byS = 1;
            }
        }
    };


    inline dim3 getDefaultBlockSize(const uint& width, const uint& height) {
        constexpr uint blockDimX[4] = { 32, 64, 128, 256 };  // Possible block sizes in the x axis
        constexpr uint blockDimY[2][4] = { { 8,  4,   2,   1},
                                          { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x

        uint minDiscardedThreads = UINT_MAX;
        uint bxS = 0; // from 0 to 3
        uint byS = 0; // from 0 to 1

        computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

        return dim3(blockDimX[bxS], blockDimY[byS][bxS]);
    }

    enum MemType { Device, Host, HostPinned };
    enum ND { _1D = 1, _2D = 2, _3D = 3, T3D = 4 };

    template <ND D>
    struct PtrDims;

    template <>
    struct PtrDims<_1D> {
        uint width{0};
        uint pitch{0};

        FK_HOST_DEVICE_CNST
            PtrDims<_1D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<_1D>(uint width_, uint pitch_ = 0) : width(width_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<_2D> {
        uint width{ 0 };
        uint height{ 0 };
        uint pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<_2D>() {}
        FK_HOST_DEVICE_CNST PtrDims<_2D>(uint width_, uint height_, uint pitch_ = 0) :
            width(width_), height(height_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<_3D> {
        // Image batch shape
        // R,G,B
        // R,G,B
        // R,G,B

        // Width and Height of one individual image
        uint width{0};
        uint height{0};
        // Number of images
        uint planes{0};
        // Number of color channels
        uint color_planes{0};

        // Pitch for each image
        uint pitch{0};

        // Pitch to jump one plane
        uint plane_pitch{0};

        FK_HOST_DEVICE_CNST PtrDims<_3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<_3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_ = 0) :
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
        uint width{ 0 };
        uint height{ 0 };
        // Number of images
        uint planes{ 0 };
        // Number of color channels
        uint color_planes{ 0 };

        // Pitch for each image
        uint pitch{ 0 };

        // Pitch to jump one plane
        uint plane_pitch{ 0 };

        // Pitch to jump to the next plane of the same image
        uint color_planes_pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<T3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<T3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(0), plane_pitch(0), color_planes_pitch(0) {}
    };

    template <int W>
    struct StaticPtrDims1D {
        static constexpr uint width{ W };
    };

    template <int W, int H>
    struct StaticPtrDims2D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
    };

    template<int W, int H, int P>
    struct StaticPtrDims3D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
        static constexpr uint planes{ P };
    };

    template <ND D, typename T>
    struct RawPtr;

    template <typename T>
    struct RawPtr<_1D, T> {
        T* data{nullptr};
        PtrDims<_1D> dims;
        using type = T;
        enum { ND = _1D };
    };

    template <typename T>
    struct RawPtr<_2D, T> {
        T* data{nullptr};
        PtrDims<_2D> dims;
        using type = T;
        enum { ND = _2D };
    };

    template <typename T>
    struct RawPtr<_3D, T> {
        T* data{nullptr};
        PtrDims<_3D> dims;
        using type = T;
        enum { ND = _3D };
    };

    template <typename T>
    struct RawPtr<T3D, T> {
        T* data{nullptr};
        PtrDims<T3D> dims;
        using type = T;
        enum { ND = T3D };
    };

    template<typename Dims, typename T>
    struct StaticRawPtr;

    template<typename T, int W>
    struct StaticRawPtr<StaticPtrDims1D<W>, T> {
        using type = T;
        T data[W];
        static constexpr StaticPtrDims1D<W> dims{};
        static constexpr ND ND{ _1D };
    };

    template<typename T, int W, int H>
    struct StaticRawPtr<StaticPtrDims2D<W, H>, T> {
        using type = T;
        T data[H][W];
        static constexpr StaticPtrDims2D<W, H> dims{};
        static constexpr ND ND{ _2D };
    };

    template<typename T, int W, int H, int P>
    struct StaticRawPtr<StaticPtrDims3D<W, H, P>, T> {
        using type = T;
        T data[P][H][W];
        static constexpr StaticPtrDims3D<W, H, P> dims{};
        static constexpr ND ND{ _3D };
    };

} // namespace fk

#endif