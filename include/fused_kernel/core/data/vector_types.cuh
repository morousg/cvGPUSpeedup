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

#include <cuda_runtime.h>

#define DEFINE_VECTOR_TYPES(TYPE) \
struct TYPE ## 8 { TYPE x, y, z, w, i, j, k, l; }; \
struct TYPE ## 12 { TYPE x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3; };

DEFINE_VECTOR_TYPES(char)
DEFINE_VECTOR_TYPES(uchar)
DEFINE_VECTOR_TYPES(short)
DEFINE_VECTOR_TYPES(ushort)
DEFINE_VECTOR_TYPES(int)
DEFINE_VECTOR_TYPES(uint)
DEFINE_VECTOR_TYPES(long)
DEFINE_VECTOR_TYPES(ulong)
DEFINE_VECTOR_TYPES(longlong)
DEFINE_VECTOR_TYPES(ulonglong)
DEFINE_VECTOR_TYPES(float)
DEFINE_VECTOR_TYPES(double)

#undef DEFINE_VECTOR_TYPES
