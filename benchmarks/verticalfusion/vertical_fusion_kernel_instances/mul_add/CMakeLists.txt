
# --- Paths ---
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/muladd/${NUM_EXPERIMENTS}_experiments") # Output directory for generated files
file(MAKE_DIRECTORY ${GENERATED_DIR}) # Create directory if it doesn't exist

add_vertical_fusion_benchmark("benchmark_vertical_fusionCompareMulAdd" "${GENERATED_DIR}" "mul_add")
# ---add target for the generated kernels---
# This is the target for the generated kernels



