#include <cuda.h>
#include <cuda_runtime.h>

// This should not go to the paper
struct UnaryType {};
struct BinaryType {};
#define S_ASSERT_INPUT_OUTPUT
#define STATIC_ASSERT_INSTANCE_TYPE
// end no to paper

// Figure operations
template <typename I, typename O>
struct UnaryOperationName {
  using InputType = I;
  using OutputType = O;
  using InstanceType = UnaryType;
  static constexpr __device__ __forceinline__
  OutputType exec(const InputType &input) {
      return /*implementation*/;
  }
};

template <typename I, typename P, typename O>
struct BinaryOperationName {
  using InputType = I;
  using ParamsType = P;
  using OutputType = O;
  using InstanceType = UnaryType;
  static constexpr __device__ __forceinline__
  OutputType exec(const InputType &input, const ParamsType &params) {
    return /*implementation*/;
  }
};
// end Figure operations

// Figure operation instances
template <typename Op> struct UnaryOperationInstance {
  using Operation = Op;
  using InstanceType = UnaryType;
  // Macro to statically check that Op::InstanceType == UnaryType
  // Generates a human readable message to better understand the issue
  STATIC_ASSERT_INSTANCE_TYPE
};

template <typename Op> struct BinaryOperationInstance {
  using Operation = Op;
  using InstanceType = BinaryType;
  STATIC_ASSERT_INSTANCE_TYPE
  typename Operation::ParamsType params;
}; 
// end Figure operation instances


// Figure pointwise
template <typename R, typename... OpIs>
constexpr __device__ __forceinline__ 
auto process_helper(const R &result, const OpIs &...instances) {
  if constexpr (sizeof...(instances) > 0) {
    return process(result, instances…);
  } else {
    return result;
  }
}

template <typename I, typename FirstOpI, typename... OpIs>
constexpr __device__ __forceinline__ 
auto process(const I &inputReg,
             const FirstOpI &instance,
             const OpIs &...instances) {
  if constexpr (/*FirstOpI::InstanceType  == UnaryType*/) {
    const auto result = FirstOpI::Operation::exec(inputReg);
    return process_helper(result, instances...);
  } else { // Is BinaryType
    const auto result =
        FirstOpI::Operation::exec(inputReg, instance.params);
    return process_helper(result, instances...);
  }
}

template <typename I, typename O, typename... OpIs>
__global__ void pointwise(const I input, 
                          O output,
                          const OpIs... instances) {
  S_ASSERT_INPUT_OUTPUT // Check that I/O types match
  int idx = 0; // compute thread idx
  const I inputReg = input[idx]; // read input for current thread
  output[idx] = process(inputReg, instances...);
}
// end Figure pointwise
