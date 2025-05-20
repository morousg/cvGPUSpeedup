
 include(intellisense.cmake)
 function(add_vertical_fusion_kernels TARGET_NAME FOLDER_FROM)
    file(
        GLOB_RECURSE
        VF_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        "${FOLDER_FROM}/*"
    )
    target_sources(${TARGET_NAME} PRIVATE ${VF_KERNEL_SOURCES})
endfunction()# --- Configuration ---


function (generate_vf_kernels NUM_EXPERIMENTS_AS_INT GENERATED_DIR GENERATED_CU_FILES_LIST OUT_LAUNCHER_H_FILE)
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

    set(OUT_LAUNCHER_H_FILE1 "${GENERATED_DIR}/mul_launcher.h")
    configure_file("${IN_LAUNCHER_H}" "${OUT_LAUNCHER_H_FILE1}" @ONLY)
    set(OUT_LAUNCHER_H_FILE "${OUT_LAUNCHER_H_FILE1}" PARENT_SCOPE)
    
endfunction()

function (add_vertical_fusion_benchmark TARGET_NAME GENERATED_DIR GENERATED_CU_FILES_LIST OUT_LAUNCHER_H_FILE)
    generate_vf_kernels("${NUM_EXPERIMENTS_AS_INT}" "${GENERATED_DIR}" "${GENERATED_CU_FILES_LIST}" "${OUT_LAUNCHER_H_FILE}")         
    add_executable(${TARGET_NAME}  launcher.cu mulOpType.cuh realBatch.h ${GENERATED_CU_FILES_LIST} "${OUT_LAUNCHER_H_FILE}"
        ${GENERATED_DIR}/mul_launcher.h)
        
    add_vertical_fusion_kernels(${TARGET_NAME}  ${GENERATED_DIR})
    add_cuda_to_target(${TARGET_NAME} "")

    # --- Include Directories ---
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include    
        ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>
        ${GENERATED_DIR}                     # For "mulN.h", "mul_launcher.h"
        
    )
    add_fkl_to_target(${TARGET_NAME})                 
    add_opencv_to_target(${TARGET_NAME} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
    enable_intellisense(${TARGET_NAME})

    
    # To pass NUM_EXPERIMENTS_AS_INT to your C++ code (e.g., main.cu)
    # In C++: int max_exp = CPP_NUM_EXPERIMENTS;
    target_compile_definitions(${TARGET_NAME} PRIVATE CPP_NUM_EXPERIMENTS=${NUM_EXPERIMENTS_AS_INT})
    
endfunction()

set(NUM_EXPERIMENTS 50 CACHE STRING "Number of experiments to generate (e.g., 50)")
option (NUM_EXPERIMENTS "Number of experiments to generate (e.g., 50)" ${NUM_EXPERIMENTS})
math(EXPR NUM_EXPERIMENTS_AS_INT "${NUM_EXPERIMENTS}") # Convert to integer for loop
if(NUM_EXPERIMENTS_AS_INT LESS_EQUAL 0)
    message(FATAL_ERROR "NUM_EXPERIMENTS must be a positive integer. Value: ${NUM_EXPERIMENTS}")
endif()
# Source directory for <benchmarks/...> includes
# Adjust this if your 'benchmarks' directory is elsewhere (e.g., inside a 'src' folder)
set(BENCHMARKS_INCLUDE_ROOT "${CMAKE_SOURCE_DIR}")

add_subdirectory(opencv/verticalfusion/vertical_fusion_kernel_instances/mul)
add_subdirectory(opencv/verticalfusion/vertical_fusion_kernel_instances/mul1)
