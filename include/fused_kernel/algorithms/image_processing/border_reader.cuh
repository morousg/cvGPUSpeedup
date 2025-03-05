/* Copyright 2025 Grup Mediapro S.L.U

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

#ifndef FK_BORDER_READER_CUH
#define FK_BORDER_READER_CUH

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.cuh>

#include <fused_kernel/core/execution_model/default_builders_def.h>
namespace fk {
    //! Border types, image boundaries are denoted with `|`
    enum class BorderType : int {
        CONSTANT = 0,    //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
        REPLICATE = 1,   //!< `aaaaaa|abcdefgh|hhhhhhh`
        REFLECT = 2,     //!< `fedcba|abcdefgh|hgfedcb`
        WRAP = 3,        //!< `cdefgh|abcdefgh|abcdefg`
        REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
        TRANSPARENT_ = 5, //!< `uvwxyz|abcdefgh|ijklmno` - Treats outliers as transparent.

        DEFAULT = REFLECT_101, //!< same as BORDER_REFLECT_101
        ISOLATED = 16 //!< Interpolation restricted within the ROI boundaries.
    };
    template<enum BorderType BT, typename ReadType>
    struct BorderReaderParameters {};
    template<typename ReadType>
    struct BorderReaderParameters<BorderType::CONSTANT, ReadType> {
        ReadType value;
    };

    template <enum BorderType BT, typename BackIOp = void, typename Enabler = void>
    struct BorderReader;
    
    template <enum BorderType BT>
    struct BorderReader<BT, void, void> {
        template <typename T, enum BorderType BT_ = BT>
        FK_HOST_FUSE
        std::enable_if_t<BT_ == BorderType::CONSTANT,
                         decltype(BorderReader<BT, TypeList<void, T>>::build(std::declval<T>()))>
        build(const T& defaultValue) {
            return BorderReader<BT, TypeList<void, T>>::build(defaultValue);
        }

        template <typename T, enum BorderType BT_ = BT>
        FK_HOST_FUSE
        std::enable_if_t<BT_ != BorderType::CONSTANT,
                         decltype(BorderReader<BT, TypeList<void, T>>::build())>
        build() {
            return BorderReader<BT, TypeList<void, T>>::build();
        }

        template <typename BF, enum BorderType BT_ = BT>
        FK_HOST_FUSE 
        std::enable_if_t<BT_ != BorderType::CONSTANT,
                         decltype(BorderReader<BT, BF>::build(std::declval<BF>()))>
        build(const BF& backFunction) {
            return BorderReader<BT, BF>::build(backFunction);
        }

        template <typename BF, enum BorderType BT_ = BT>
        FK_HOST_FUSE 
        std::enable_if_t<BT_ == BorderType::CONSTANT,
                         decltype(BorderReader<BT, BF>::build(std::declval<BF>()))>
        build(const BF& backFunction,
              const typename BF::Operation::ReadDataType& defaultValue) {
            return BorderReader<BT, BF>::build(backFunction, defaultValue);
        }
    };

    template <enum BorderType BT, typename T>
    struct BorderReader<BT, TypeList<void, T>> {
        using ReadDataType = T;
        using OutputType = T;
        using ParamsType = BorderReaderParameters<BT, ReadDataType>;
        using BackFunction = int;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<BorderReader<BT, TypeList<void, T>>>;
        
        using InstantiableType = ReadBack<BorderReader<BT, TypeList<void, T>>>;
        DEFAULT_BUILD
        template <typename ReadIOp>
        FK_HOST_FUSE auto build(const ReadIOp& readIOp, const InstantiableType& iOp) {
            return BorderReader<BT, ReadIOp>::build(iOp.params, readIOp);
        }
        DEFAULT_READ_BATCH_BUILD
    };

#define BODER_READER_DETAILS(BT) \
using ReadDataType = typename BackIOp::Operation::OutputType; \
using OutputType = ReadDataType; \
using ParamsType = BorderReaderParameters<BT, ReadDataType>; \
using BackFunction = BackIOp; \
using InstanceType = ReadBackType; \
using OperationDataType = OperationData<BorderReader<BT, BackFunction>>;

#define BODER_READER_EXEC \
FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) { \
    const int last_col = BackFunction::Operation::num_elems_x(thread, opData) - 1; \
    const int last_row = BackFunction::Operation::num_elems_y(thread, opData) - 1; \
    const Point new_thread(idx_col(thread.x, last_col), idx_row(thread.y, last_row), thread.z); \
    return BackFunction::Operation::exec(new_thread, opData.back_function); \
}

    template <typename BackIOp>
    struct BorderReader<BorderType::CONSTANT, BackIOp,
                        std::enable_if_t<!std::is_same_v<BackIOp, void> &&
                                         !std::is_same_v<BackIOp, TypeList<void, typename BackIOp::Operation::ReadDataType>>, void>> {
        BODER_READER_DETAILS(BorderType::CONSTANT)
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            if (thread.x >= 0 && thread.x < width && thread.y >= 0 && thread.y < height) {
                return BackFunction::Operation::exec(thread, opData.back_function);
            } else {
                return opData.params.value;
            }
        }

        using InstantiableType = ReadBack<BorderReader<BorderType::CONSTANT, BackIOp>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS_BACKIOP
        FK_HOST_FUSE auto build(const BackFunction& backFunction, const ReadDataType& defaultValue) {
            return InstantiableType{ OperationDataType{{defaultValue}, backFunction} };
        }
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename BackIOp>
    struct BorderReader<BorderType::REPLICATE, BackIOp,
                        std::enable_if_t<!std::is_same_v<BackIOp, void>, void>> {
        BODER_READER_DETAILS(BorderType::REPLICATE)
        BODER_READER_EXEC
        
        using InstantiableType = ReadBack<BorderReader<BorderType::REPLICATE, BackIOp>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS_BACKIOP
        FK_HOST_FUSE auto build(const BackFunction& backFunction) {
            return InstantiableType{ OperationDataType{{}, backFunction} };
        }
        DEFAULT_READ_BATCH_BUILD
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y) {
            return cxp::max(y, 0);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return cxp::min(y, last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row));
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x) {
            return cxp::max(x, 0);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return cxp::min(x, last_col);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col));
        }
    };

    template <typename BackIOp>
    struct BorderReader<BorderType::REFLECT, BackIOp, std::enable_if_t<!std::is_same_v<BackIOp, void>, void>> {
        BODER_READER_DETAILS(BorderType::REFLECT)
        BODER_READER_EXEC

        using InstantiableType = ReadBack<BorderReader<BorderType::REFLECT, BackIOp>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS_BACKIOP
        FK_HOST_FUSE auto build(const BackFunction& backFunction) {
            return InstantiableType{ OperationDataType{{}, backFunction} };
        }
        DEFAULT_READ_BATCH_BUILD
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& last_row) {
            return (cxp::abs(y) - (y < 0)) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return (last_row - cxp::abs(last_row - y) + (y > last_row));
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row), last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& last_col) {
            return (cxp::abs(x) - (x < 0)) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return (last_col - cxp::abs(last_col - x) + (x > last_col));
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col), last_col);
        }
    };

    template <typename BackIOp>
    struct BorderReader<BorderType::WRAP, BackIOp, std::enable_if_t<!std::is_same_v<BackIOp, void>, void>> {
        BODER_READER_DETAILS(BorderType::WRAP)
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            const int width = BackFunction::Operation::num_elems_x(thread, opData);
            const int height = BackFunction::Operation::num_elems_y(thread, opData);
            const Point new_thread(idx_col(thread.x, width), idx_row(thread.y, height), thread.z);
            return BackFunction::Operation::exec(new_thread, opData.back_function);
        }

        using InstantiableType = ReadBack<BorderReader<BorderType::WRAP, BackIOp>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS_BACKIOP
        FK_HOST_FUSE auto build(const BackFunction& backFunction) {
            return InstantiableType{ OperationDataType{{}, backFunction} };
        }
        DEFAULT_READ_BATCH_BUILD
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& height) {
            return (y >= 0) ? y : (y - ((y - height + 1) / height) * height);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& height) {
            return (y < height) ? y : (y % height);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& height) {
            return idx_row_low(idx_row_high(y, height), height);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& width) {
            return (x >= 0) ? x : (x - ((x - width + 1) / width) * width);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& width) {
            return (x < width) ? x : (x % width);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& width) {
            return idx_col_low(idx_col_high(x, width), width);
        }
    };

    template <typename BackIOp>
    struct BorderReader<BorderType::REFLECT_101, BackIOp, std::enable_if_t<!std::is_same_v<BackIOp, void>, void>> {
        BODER_READER_DETAILS(BorderType::REFLECT_101)
        BODER_READER_EXEC

        using InstantiableType = ReadBack<BorderReader<BorderType::REFLECT_101, BackIOp>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS_BACKIOP
        FK_HOST_FUSE auto build(const BackFunction& backFunction) {
            return InstantiableType{ OperationDataType{{}, backFunction} };
        }
        DEFAULT_READ_BATCH_BUILD
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& last_row) {
            return cxp::abs(y) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return cxp::abs(last_row - cxp::abs(last_row - y)) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row), last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& last_col) {
            return cxp::abs(x) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return cxp::abs(last_col - cxp::abs(last_col - x)) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col), last_col);
        }
    };

#undef BODER_READER_DETAILS
#undef BODER_READER_EXEC
#include <fused_kernel/core/execution_model/default_builders_undef.h>

}

#endif // FK_BORDER_READER_CUH