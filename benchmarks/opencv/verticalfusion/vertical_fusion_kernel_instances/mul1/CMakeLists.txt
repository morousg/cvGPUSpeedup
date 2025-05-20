
# --- Paths ---
set(TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates") # Store your .in files here
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/mul1/${NUM_EXPERIMENTS}_experiments") # Output directory for generated files
file(MAKE_DIRECTORY ${GENERATED_DIR}) # Create directory if it doesn't exist

# --- Template File Names ---
set(IN_KERNEL_H "${TEMPLATE_DIR}/experiment_kernel.h.in")
set(IN_KERNEL_CU "${TEMPLATE_DIR}/experiment_kernel.cu.in")
set(IN_LAUNCHER_H "${TEMPLATE_DIR}/experiment_launcher.h.in") # Main dispatcher template

# --- File Generation Loop ---
set(GENERATED_CU_FILES_LIST "")    # To collect all mulN.cu files for the executable
set(LAUNCHER_INCLUDES_BLOCK "")    # To build the #include block for mul_launcher.h
set(LAUNCHER_DISPATCH_BLOCK "") # To build the DISPATCH_TO_MUL_INSTANCE block




add_vertical_fusion_benchmark("benchmark_vertical_fusionCompareMul1" "${GENERATED_DIR}" "${GENERATED_CU_FILES_LIST}" "${OUT_LAUNCHER_H_FILE}")
# ---add target for the generated kernels---
# This is the target for the generated kernels



