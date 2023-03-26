#pragma once

#include "fast_kernel.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// This is a prototype interface to my kernel fusion library, to test
// OpenCV programmer's opinion on the shape of it.

namespace cvGS {

template <typename I, typename O>
unary_operation_scalar<unary_cuda_vector_cast<I, O>, I, O> convertTo() {
    return {};
}

// This are just a quick mockup of the future generic functions
// They only work with types that have 3 components. In the future
// they will work with anything.
template <typename I, typename O>
binary_operation_scalar<binary_mul<I, O>, I, O> multiply(cv::Scalar src2) {
    return {make_<O>(src2[0], src2[1], src2[2])};
}

template <typename I, typename O>
binary_operation_scalar<binary_sub<I>, I, O> subtract(cv::Scalar src2) {
    return {make_<O>(src2[0], src2[1], src2[2])};
}

template <typename I, typename O>
binary_operation_scalar<binary_div<I>, I, O> divide(cv::Scalar src2) {
    return {make_<O>(src2[0], src2[1], src2[2])};
}

template <typename I>
split_write_scalar<perthread_split_write<I>, I> split(std::vector<cv::cuda::GpuMat>& output) {
    return {(float*)output.at(0).data, (float*)output.at(1).data, (float*)output.at(2).data};
}

// Need to find a way to convert CV types to cuda vector types
// If we do it by reading GpuMat types, we will need execution time
// branching to select the call with the proper type. That is tedious work,
// I want to find a way to generate this code.
// An alternative, is to directly ask the programmer for the CV type, and to pass it
// as a template type. This way, we can find the corresponding types in compile time
// and generate a more efficient CPU code.
template <typename... operations>
void executeOperations(cv::cuda::GpuMat& input, cv::cuda::Stream& stream, operations... ops) {
    int num_elems = input.rows * input.cols;
    dim3 block(256);
    dim3 grid(ceil(num_elems / (float)block.x));
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    cuda_transform_noret<<<grid, block, 0, cu_stream>>>(num_elems, (uchar3*)input.data, ops...);
    gpuErrchk(cudaGetLastError());
}

} // namespace cvGS
