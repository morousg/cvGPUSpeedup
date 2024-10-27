#include <cuda.h>
#include <cuda_runtime.h>

struct UnaryType {};
struct BinaryType {};

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

#define S_ASSERT_INPUT_OUTPUT

template <typename R, typename... OpIs>
constexpr __device__ __forceinline__ auto process_helper(const R &result, const OpIs &...instances) {
  if constexpr (sizeof…(instances) > 0) {
    return process(result, instances…);
  } else {
    return result;
  }
}

template <typename I, typename FirstOpI, typename... OpIs>
constexpr __device__ __forceinline__ auto process(const I &inputReg, const FirstOpI &instance,
                                                  const OpIs &...instances) {
  if constexpr (/*FirstOpI::InstanceType  == UnaryType*/) {
    const auto result = FirstOpI::Operation::exec(inputReg);
    return process_helper(result, instances…);
  } else { // Is BinaryType
    const auto result = FirstOpI::Operation::exec(inputReg, instance.params);
    return process_helper(result, instances…);
  }
}

template <typename I, typename O, typename... OpIs>
__global__ void pointwise(const I input, O output, const OpIs... instances) {
  S_ASSERT_INPUT_OUTPUT     // Check that I/O types match
      const I inputReg = 0; // read input for current thread
  output[index] = process(inputReg, instances…);
}
