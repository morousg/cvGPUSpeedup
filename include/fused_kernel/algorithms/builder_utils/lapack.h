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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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

/* Copyright 2025 Oscar Amoros Huguet
*  Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Changes performed over the original OpenCV code will be marked with the comment // Change: explanation

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/data/point.h>
#include <cassert>

namespace fk {
    // Change: the enum DecompTypes now belongs to the namespace fk and the comment for DECOMP_NORMAL has been shortened
    //! matrix decomposition types
    enum class DecompTypes : int {
        /** Gaussian elimination with the optimal pivot element chosen. */
        DECOMP_LU = 0,
        /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
        src1 can be singular */
        DECOMP_SVD = 1,
        /** eigenvalue decomposition; the matrix src1 must be symmetrical */
        DECOMP_EIG = 2,
        /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
        defined */
        DECOMP_CHOLESKY = 3,
        /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
        DECOMP_QR = 4,
        /** while all the previous flags are mutually exclusive, this flag can be used together with
        any of the previous; */
        DECOMP_NORMAL = 16
    };

#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))
#define Sf( y, x ) ((float*)(srcdata + y*srcstep))[x]
#define Sd( y, x ) ((double*)(srcdata + y*srcstep))[x]
#define Df( y, x ) ((float*)(dstdata + y*dststep))[x]
#define Dd( y, x ) ((double*)(dstdata + y*dststep))[x]

    /** @brief Aligns a buffer size to the specified number of bytes.

    The function returns the minimum number that is greater than or equal to sz and is divisible by n :
    \f[\texttt{(sz + n-1) & -n}\f]
    @param sz Buffer size to align.
    @param n Alignment size that must be a power of two.
     */
    static inline size_t alignSize(size_t sz, int n) {
        if ((n & (n - 1)) != 0) {
            throw std::runtime_error("n must be a power of 2");
        } // n is a power of 2
        return (sz + n - 1) & -n;
    }

    /** @brief Aligns a pointer to the specified number of bytes.

    The function returns the aligned pointer of the same type as the input pointer:
    \f[\texttt{(_Tp*)(((size_t)ptr + n-1) & -n)}\f]
    @param ptr Aligned pointer.
    @param n Alignment size that must be a power of two.
     */
    template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp)) {
        if ((n & (n - 1)) != 0) {
            throw std::runtime_error("n must be a power of 2");
        } // n is a power of 2
        return (_Tp*)(((size_t)ptr + n - 1) & -n);
    }

    template <typename PtrType>
    static constexpr void mulTransposed(const PtrType& src, PtrType& dst, bool aTa) {
        using SType = typename PtrType::Type;
        using DType = SType;
        static_assert(std::is_same_v<SType, float> || std::is_same_v<SType, double>, "Ptr types must be float or double");

        DType* dptr = dst.ptr().data;
        size_t dstep = dst.step / sizeof(dptr[0]);
        size_t sstep = src.step / sizeof(src.ptr<_Tp>()[0]);
        int i, j, k, n = aTa ? src.cols : src.rows, m = aTa ? src.rows : src.cols;
        double delta_buf[4] = { 0, 0, 0, 0 };

        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                double s = 0;
                const DType* sptr1 = src.ptr().data + sstep * (aTa ? 0 : i);
                const DType* sptr2 = src.ptr().data + sstep * (aTa ? 0 : j);

                for (k = 0; k < m; k++) {
                    s += (double)sptr1[k] * sptr2[k];
                }

                dptr[i * dstep + j] = (_Tp)s;
                if (i != j)
                    dptr[j * dstep + i] = (_Tp)s;
            }
        }
    }

    // Change: Types InputArray and OutputArray are now template types, following the Static Polimorphism of
    // the FK library.
    template <typename InputArray1, typename InputArray2, typename OutputArray>
    constexpr inline bool solve(const InputArray1& src, const InputArray2& _src2, OutputArray& dst, const DecompTypes& method) {
        // Change: CV_INSTRUMENT_REGION(); removed

        bool result = true;
        // Change: not using Mat, no need to do this line Mat src = _src.getMat(), _src2 = _src2arg.getMat();
        using TypeI1 = typename InputArray1::Type;
        using TypeI2 = typename InputArray2::Type;
        using TypeO = typename OutputArray::Type;
        static_assert(std::is_same_v<TypeI1, TypeI2> && std::is_same_v<TypeI2, TypeO>, "All types must be the same");
        static_assert(std::is_same_v<TypeI1, float> || std::is_same_v<TypeI1, double>, "Input and output type must be float or double");
        bool is_normal = (static_cast<uint>(method) & static_cast<uint>(DecompTypes::DECOMP_NORMAL)) != 0;

        DecompTypes method2 = static_cast<DecompTypes>((~static_cast<uint>(DecompTypes::DECOMP_NORMAL)) & static_cast<uint>(method));
        assert(method2 == DecompTypes::DECOMP_LU || method2 == DecompTypes::DECOMP_SVD || method2 == DecompTypes::DECOMP_EIG ||
            method2 == DecompTypes::DECOMP_CHOLESKY || method2 == DecompTypes::DECOMP_QR &&
            "Unsupported method, see #DecompTypes");
        assert((method2 != DecompTypes::DECOMP_LU && method2 != DecompTypes::DECOMP_CHOLESKY) || is_normal || src.dims().height == src.dims().width);

        // check case of a single equation and small matrix
        if ((method2 == DecompTypes::DECOMP_LU || method2 == DecompTypes::DECOMP_CHOLESKY) && !is_normal &&
            src.dims().height <= 3 && src.dims().height == src.dims().width && _src2.dims().width == 1) {
#define bf(y) ((float*)(bdata + y*src2step))[0]
#define bd(y) ((double*)(bdata + y*src2step))[0]

            const uchar* srcdata = reinterpret_cast<uchar*>(src.ptr().data);
            const uchar* bdata = reinterpret_cast<uchar*>(_src2.ptr().data);
            uchar* dstdata = reinterpret_cast<uchar*>(dst.ptr().data);
            size_t srcstep = src.dims().width * sizeof(typename InputArray1::Type);
            size_t src2step = _src2.dims().width * sizeof(typename InputArray2::Type);
            size_t dststep = dst.dims().width * sizeof(typename OutputArray::Type);

            if (src.dims().height == 2) {
                if constexpr (std::is_same_v<typename InputArray1::Type, float>) {
                    double d = det2(Sf);
                    if (d != 0.) {
                        double t;
                        d = 1. / d;
                        t = (float)(((double)bf(0) * Sf(1, 1) - (double)bf(1) * Sf(0, 1)) * d);
                        Df(1, 0) = (float)(((double)bf(1) * Sf(0, 0) - (double)bf(0) * Sf(1, 0)) * d);
                        Df(0, 0) = (float)t;
                    } else {
                        result = false;
                    }
                } else {
                    double d = det2(Sd);
                    if (d != 0.) {
                        double t;
                        d = 1. / d;
                        t = (bd(0) * Sd(1, 1) - bd(1) * Sd(0, 1)) * d;
                        Dd(1, 0) = (bd(1) * Sd(0, 0) - bd(0) * Sd(1, 0)) * d;
                        Dd(0, 0) = t;
                    } else {
                        result = false;
                    }
                }
            } else if (src.dims().height == 3) {
                if constexpr (std::is_same_v<typename InputArray1::Type, float>) {
                    double d = det3(Sf);
                    if (d != 0.) {
                        float t[3];
                        d = 1. / d;

                        t[0] = (float)(d *
                            (bf(0) * ((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) -
                                Sf(0, 1) * ((double)bf(1) * Sf(2, 2) - (double)Sf(1, 2) * bf(2)) +
                                Sf(0, 2) * ((double)bf(1) * Sf(2, 1) - (double)Sf(1, 1) * bf(2))));

                        t[1] = (float)(d *
                            (Sf(0, 0) * (double)(bf(1) * Sf(2, 2) - (double)Sf(1, 2) * bf(2)) -
                                bf(0) * ((double)Sf(1, 0) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 0)) +
                                Sf(0, 2) * ((double)Sf(1, 0) * bf(2) - (double)bf(1) * Sf(2, 0))));

                        t[2] = (float)(d *
                            (Sf(0, 0) * ((double)Sf(1, 1) * bf(2) - (double)bf(1) * Sf(2, 1)) -
                                Sf(0, 1) * ((double)Sf(1, 0) * bf(2) - (double)bf(1) * Sf(2, 0)) +
                                bf(0) * ((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0))));

                        Df(0, 0) = t[0];
                        Df(1, 0) = t[1];
                        Df(2, 0) = t[2];
                    } else {
                        result = false;
                    }
                    
                } else {
                    double d = det3(Sd);
                    if (d != 0.) {
                        double t[9];

                        d = 1. / d;

                        t[0] = ((Sd(1, 1) * Sd(2, 2) - Sd(1, 2) * Sd(2, 1)) * bd(0) +
                            (Sd(0, 2) * Sd(2, 1) - Sd(0, 1) * Sd(2, 2)) * bd(1) +
                            (Sd(0, 1) * Sd(1, 2) - Sd(0, 2) * Sd(1, 1)) * bd(2)) * d;

                        t[1] = ((Sd(1, 2) * Sd(2, 0) - Sd(1, 0) * Sd(2, 2)) * bd(0) +
                            (Sd(0, 0) * Sd(2, 2) - Sd(0, 2) * Sd(2, 0)) * bd(1) +
                            (Sd(0, 2) * Sd(1, 0) - Sd(0, 0) * Sd(1, 2)) * bd(2)) * d;

                        t[2] = ((Sd(1, 0) * Sd(2, 1) - Sd(1, 1) * Sd(2, 0)) * bd(0) +
                            (Sd(0, 1) * Sd(2, 0) - Sd(0, 0) * Sd(2, 1)) * bd(1) +
                            (Sd(0, 0) * Sd(1, 1) - Sd(0, 1) * Sd(1, 0)) * bd(2)) * d;

                        Dd(0, 0) = t[0];
                        Dd(1, 0) = t[1];
                        Dd(2, 0) = t[2];
                    } else {
                        result = false;
                    }
                }
            } else {
                if (src.dims().height != 1) {
                    throw std::runtime_error("Wrong height in src parameter");
                }

                if constexpr (std::is_same_v<typename InputArray1::Type, float>) {
                    double d = Sf(0, 0);
                    if (d != 0.)
                        Df(0, 0) = (float)(bf(0) / d);
                    else
                        result = false;
                } else {
                    double d = Sd(0, 0);
                    if (d != 0.)
                        Dd(0, 0) = (bd(0) / d);
                    else
                        result = false;
                }
            }
            return result;
        }

        int m = src.dims().height, m_ = m, n = src.dims().width, nb = _src2.dims().width;
        size_t esz = sizeof(InputArray1), bufsize = 0;
        size_t vstep = alignSize(n * esz, 16);
        size_t astep = method2 == DecompTypes::DECOMP_SVD && !is_normal ? alignSize(m * esz, 16) : vstep;
        Ptr1D<uchar> buffer;

        //_dst.create(src.cols, src2.cols, src.type());
        //Mat dst = _dst.getMat();

        if (m < n) {
            throw std::runtime_error("The function can not solve under-determined linear systems");
        } else if (m == n) {
            is_normal = false;
        } else if (is_normal) {
            m_ = n;
            if (method2 == DecompTypes::DECOMP_SVD) {
                method2 = DecompTypes::DECOMP_EIG;
            }
        }

        size_t asize = astep * (method2 == DecompTypes::DECOMP_SVD || is_normal ? n : m);
        bufsize += asize + 32;

        if (is_normal) {
            bufsize += n * nb * esz;
        }
        if (method2 == DecompTypes::DECOMP_SVD || method2 == DecompTypes::DECOMP_EIG) {
            bufsize += n * 5 * esz + n * vstep + nb * sizeof(double) + 32;
        }

        buffer.allocPtr(PtrDims<_1D>{static_cast<uint>(bufsize)}, MemType::Host);
        uchar* ptr = alignPtr(buffer.ptr().data, 16);

        Ptr2D<TypeI1> a(reinterpret_cast<TypeI1*>(ptr), n, m_, astep, MemType::Host);

        if (is_normal) {
            //mulTransposed(src, a, true);
        } else if (method2 != DecompTypes::DECOMP_SVD) {
            //src.copyTo(a);
        } else {
            //a = Mat(n, m_, type, ptr, astep);
            //transpose(src, a);
        }
        ptr += asize;

        /*if (!is_normal) {
            if (method2 == DECOMP_LU || method2 == DECOMP_CHOLESKY)
                src2.copyTo(dst);
        } else {
            // a'*b
            if (method2 == DECOMP_LU || method2 == DECOMP_CHOLESKY)
                gemm(src, src2, 1, Mat(), 0, dst, GEMM_1_T);
            else {
                Mat tmp(n, nb, type, ptr);
                ptr += n * nb * esz;
                gemm(src, src2, 1, Mat(), 0, tmp, GEMM_1_T);
                src2 = tmp;
            }
        }

        if (method2 == DECOMP_LU) {
            if (type == CV_32F)
                result = hal::LU32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb) != 0;
            else
                result = hal::LU64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb) != 0;
        } else if (method2 == DECOMP_CHOLESKY) {
            if (type == CV_32F)
                result = hal::Cholesky32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb);
            else
                result = hal::Cholesky64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb);
        } else if (method2 == DECOMP_QR) {
            Mat rhsMat;
            if (is_normal || m == n) {
                src2.copyTo(dst);
                rhsMat = dst;
            } else {
                rhsMat = Mat(m, nb, type);
                src2.copyTo(rhsMat);
            }

            if (type == CV_32F)
                result = hal::QR32f(a.ptr<float>(), a.step, a.rows, a.cols, rhsMat.cols, rhsMat.ptr<float>(), rhsMat.step, NULL) != 0;
            else
                result = hal::QR64f(a.ptr<double>(), a.step, a.rows, a.cols, rhsMat.cols, rhsMat.ptr<double>(), rhsMat.step, NULL) != 0;

            if (rhsMat.rows != dst.rows)
                rhsMat.rowRange(0, dst.rows).copyTo(dst);
        } else {
            ptr = alignPtr(ptr, 16);
            Mat v(n, n, type, ptr, vstep), w(n, 1, type, ptr + vstep * n), u;
            ptr += n * (vstep + esz);

            if (method2 == DECOMP_EIG) {
                if (type == CV_32F)
                    Jacobi(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, n, ptr);
                else
                    Jacobi(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, n, ptr);
                u = v;
            } else {
                if (type == CV_32F)
                    JacobiSVD(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, m_, n);
                else
                    JacobiSVD(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, m_, n);
                u = a;
            }

            if (type == CV_32F) {
                SVBkSb(m_, n, w.ptr<float>(), 0, u.ptr<float>(), u.step, true,
                    v.ptr<float>(), v.step, true, src2.ptr<float>(),
                    src2.step, nb, dst.ptr<float>(), dst.step, ptr);
            } else {
                SVBkSb(m_, n, w.ptr<double>(), 0, u.ptr<double>(), u.step, true,
                    v.ptr<double>(), v.step, true, src2.ptr<double>(),
                    src2.step, nb, dst.ptr<double>(), dst.step, ptr);
            }
            result = true;
        }

        if (!result)
            dst = Scalar(0);
            
        return result;*/

        return true;
    }

#undef det2
#undef det3
#undef Sf
#undef Sd
#undef Df
#undef Dd
#undef bf
#undef bd

} // namespace fk
