

function(add_vertical_fusion_kernels TARGET_NAME FOLDER_FROM)
    file(
        GLOB_RECURSE
        VF_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        "${FOLDER_FROM}/*"
    )
    target_sources(${TARGET_NAME} PRIVATE ${VF_KERNEL_SOURCES})
endfunction()


# --- Configuration ---
set(NUM_EXPERIMENTS 50 CACHE STRING "Number of experiments to generate (e.g., 50)")
option (NUM_EXPERIMENTS "Number of experiments to generate (e.g., 50)" ${NUM_EXPERIMENTS})
math(EXPR NUM_EXPERIMENTS_AS_INT "${NUM_EXPERIMENTS}") # Convert to integer for loop
if(NUM_EXPERIMENTS_AS_INT LESS_EQUAL 0)
    message(FATAL_ERROR "NUM_EXPERIMENTS must be a positive integer. Value: ${NUM_EXPERIMENTS}")
endif()

# --- Paths ---
set(TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates") # Store your .in files here
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/mul1/${NUM_EXPERIMENTS}_experiments") # Output directory for generated files
file(MAKE_DIRECTORY ${GENERATED_DIR}) # Create directory if it doesn't exist

# Source directory for <benchmarks/...> includes
# Adjust this if your 'benchmarks' directory is elsewhere (e.g., inside a 'src' folder)
set(BENCHMARKS_INCLUDE_ROOT "${CMAKE_SOURCE_DIR}")

# --- Template File Names ---
set(IN_KERNEL_H "${TEMPLATE_DIR}/experiment_kernel.h.in")
set(IN_KERNEL_CU "${TEMPLATE_DIR}/experiment_kernel.cu.in")
set(IN_LAUNCHER_H "${TEMPLATE_DIR}/experiment_launcher.h.in") # Main dispatcher template

# --- File Generation Loop ---
set(GENERATED_CU_FILES_LIST "")    # To collect all mulN.cu files for the executable
set(LAUNCHER_INCLUDES_BLOCK "")    # To build the #include block for mul_launcher.h
set(LAUNCHER_DISPATCH_BLOCK "") # To build the DISPATCH_TO_MUL_INSTANCE block

foreach(IDX RANGE 1 ${NUM_EXPERIMENTS_AS_INT})
    set(N ${IDX}) # Set the variable 'N' that @N@ will be replaced with

    set(CURRENT_MUL_H "${GENERATED_DIR}/mul${N}.h")
    set(CURRENT_MUL_CU "${GENERATED_DIR}/mul${N}.cu")

    # Generate mulN.h
    configure_file("${IN_KERNEL_H}" "${CURRENT_MUL_H}" @ONLY)

    # Generate mulN.cu
    configure_file("${IN_KERNEL_CU}" "${CURRENT_MUL_CU}" @ONLY)

    list(APPEND GENERATED_CU_FILES_LIST ${CURRENT_MUL_CU})

    # Append to strings for the main launcher header
    string(APPEND LAUNCHER_INCLUDES_BLOCK "#include \"mul${N}.h\"\n")
    string(APPEND LAUNCHER_DISPATCH_BLOCK  "DISPATCH_TO_MUL_INSTANCE(${N})\n")
endforeach()

# --- Generate the Main Dispatcher Header (mul_launcher.h) ---
set(GENERATED_MUL_INCLUDES ${LAUNCHER_INCLUDES_BLOCK})
set(GENERATED_DISPATCH_CALLS ${LAUNCHER_DISPATCH_BLOCK})
set(NUM_EXPERIMENTS_TOTAL ${NUM_EXPERIMENTS_AS_INT}) # For static_assert in template

set(OUT_LAUNCHER_H_FILE "${GENERATED_DIR}/mul_launcher.h")
configure_file("${IN_LAUNCHER_H}" "${OUT_LAUNCHER_H_FILE}" @ONLY)


# ---add target for the generated kernels---
# This is the target for the generated kernels

set (cuda_target benchmark_vertical_fusionCompareMul1) # Name of the target 
add_executable(${cuda_target}  launcher.cu mulOpType.cuh realBatch.h ${GENERATED_CU_FILES_LIST} "${OUT_LAUNCHER_H_FILE}"
    ${GENERATED_DIR}/mul_launcher.h)
target_sources(${cuda_target} PRIVATE ${LAUNCH_SOURCES})            

add_vertical_fusion_kernels(${cuda_target}  ${GENERATED_DIR})
add_cuda_to_target(${cuda_target} "")

# --- Include Directories ---
target_include_directories(${cuda_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include    
    ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>
    ${GENERATED_DIR}                     # For "mulN.h", "mul_launcher.h"
     
)
 
# To pass NUM_EXPERIMENTS_AS_INT to your C++ code (e.g., main.cu)

target_compile_definitions(${cuda_target} PRIVATE CPP_NUM_EXPERIMENTS=${NUM_EXPERIMENTS_AS_INT})
add_fkl_to_target(${cuda_target})                 
add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
enable_intellisense(${cuda_target})

# In C++: int max_exp = CPP_NUM_EXPERIMENTS;

