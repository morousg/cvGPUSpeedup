/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef FK_ARRAY
#define FK_ARRAY

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <cstddef>

namespace fk {
    template <typename T, size_t SIZE>
    union Array {
        enum { size = SIZE };
        T at[SIZE];
        FK_HOST_DEVICE_CNST Array(const T& initValue) {
            for (int i = 0; i < static_cast<int>(SIZE); i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST Array(const std::initializer_list<T>& initList) {
            int i = 0;
            for (const auto& value : initList) {
                at[i++] = value;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const int& index) const {
            return at[index];
        }
    };

    template <typename T>
    union Array<T, 1> {
        enum { size = 1 };
        struct {
            T x;
        };
        T at[size];
        FK_HOST_DEVICE_CNST Array(const T& initValue) {
            at[0] = initValue;
        }
        FK_HOST_DEVICE_CNST Array(const std::initializer_list<T>& initList) {
            int i = 0;
            for (const auto& value : initList) {
                at[i++] = value;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const int& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return x;
        }
    };

    template <typename T>
    union Array<T, 2> {
        enum { size = 2 };
        struct {
            T x, y;
        };
        T at[size];
        FK_HOST_DEVICE_CNST Array(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST Array(const std::initializer_list<T>& initList) {
            int i = 0;
            for (const auto& value : initList) {
                at[i++] = value;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const int& index) const {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return VectorAt(index, make_<VectorType_t<T,size>>(x,y));
        }
    };

    template <typename T>
    union Array<T, 3> {
        enum { size = 3 };
        struct {
            T x, y, z;
        };
        T at[size];
        FK_HOST_DEVICE_CNST Array(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST Array(const std::initializer_list<T>& initList) {
            int i = 0;
            for (const auto& value : initList) {
                at[i++] = value;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const int& index) const {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return VectorAt(index, make_<VectorType_t<T, size>>(x, y, z));
        }
    };

    template <typename T>
    union Array<T, 4> {
        enum { size = 4 };
        struct {
            T x, y, z, w;
        };
        T at[size];
        FK_HOST_DEVICE_CNST Array(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST Array(const std::initializer_list<T>& initList) {
            int i = 0;
            for (const auto& value : initList) {
                at[i++] = value;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const int& index) const {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return VectorAt(index, make_<VectorType_t<T, size>>(x, y, z, w));
        }
    };

    template <typename V, int... Idx>
    FK_HOST_DEVICE_CNST Array<VBase<V>, cn<V>> toArray_helper(const V& vector, const std::integer_sequence<int, Idx...>&) {
        return { { VectorAt<Idx>(vector)... } };
    }

    template <typename V>
    FK_HOST_DEVICE_CNST Array<VBase<V>, cn<V>> toArray(const V& vector) {
        if constexpr (validCUDAVec<V>) {
            return toArray_helper(vector, std::make_integer_sequence<int, cn<V>>{});
        } else {
            return Array<V, 1>{ {vector} };
        }
    }

    template <typename T, int SIZE, int... Idx>
    FK_HOST_DEVICE_CNST VectorType_t<T, SIZE> toVector_helper(const Array<T, SIZE>& array_v, const std::integer_sequence<int, Idx...>&) {
        return { array_v.at[Idx]... };
    }

    template <typename T, int SIZE>
    FK_HOST_DEVICE_CNST VectorType_t<T, SIZE> toVector(const Array<T, SIZE>& array_v) {
        static_assert(SIZE <= 4, "No Vector types available with size greater than 4");
        if constexpr (SIZE == 1) {
            return array_v.at[0];
        } else {
            return toVector_helper(array_v, std::make_integer_sequence<int, SIZE>{});
        }
    }

    template <typename T, size_t BATCH, typename... Types>
    FK_HOST_DEVICE_CNST Array<T, BATCH> make_array(const Types&... pars) {
        static_assert(sizeof...(Types) == BATCH, "Too many or too few elements for the array size.");
        static_assert(std::disjunction_v<std::is_same<T, Types>...>, "All the types should be the same");
        return { {pars...} };
    }

    template <typename Array, typename Value, size_t... Idx>
    FK_HOST_DEVICE_CNST Array make_set_array_helper(const std::index_sequence<Idx...>&, const Value& value) {
        return { { (static_cast<void>(Idx), value)... } };
    }

    template <typename T, size_t BATCH>
    FK_HOST_DEVICE_CNST Array<T, BATCH> make_set_array(const T& value) {
        return make_set_array_helper<Array<T, BATCH>>(std::make_index_sequence<BATCH>(), value);
    }

    template <typename Value, size_t... Idx>
    FK_HOST_DEVICE_CNST std::array<Value, sizeof...(Idx)> make_set_std_array_helper(const std::index_sequence<Idx...>&, const Value& value) {
        return { { (static_cast<void>(Idx), value)... } };
    }

    template <size_t BATCH, typename T>
    FK_HOST_DEVICE_CNST std::array<T, BATCH> make_set_std_array(const T& value) {
        return make_set_std_array_helper(std::make_index_sequence<BATCH>(), value);
    }

    template <typename StdArray>
    struct StdArrayDetails;

    template <typename ArrayType, size_t SIZE>
    struct StdArrayDetails<std::array<ArrayType, SIZE>> {
        static constexpr size_t size{ SIZE };
        using type = ArrayType;
    };

    template <size_t BATCH, typename... ArrayTypes>
    constexpr bool allArraysSameSize_v = and_v<(StdArrayDetails<ArrayTypes>::size == BATCH)...>;

} // namespace fk

#endif
