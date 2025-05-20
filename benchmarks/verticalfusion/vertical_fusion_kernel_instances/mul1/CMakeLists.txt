
# --- Paths ---
set(TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates") # Store your .in files here
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/mul1/${NUM_EXPERIMENTS}_experiments") # Output directory for generated files
file(MAKE_DIRECTORY ${GENERATED_DIR}) # Create directory if it doesn't exist

 
 
add_vertical_fusion_benchmark("benchmark_vertical_fusionCompareMul1" "${GENERATED_DIR}" "mulOpType")
 


