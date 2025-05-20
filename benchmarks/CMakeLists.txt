
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

set(NUM_EXPERIMENTS 50 CACHE STRING "Number of experiments to generate (e.g., 50)")
option (NUM_EXPERIMENTS "Number of experiments to generate (e.g., 50)" ${NUM_EXPERIMENTS})
math(EXPR NUM_EXPERIMENTS_AS_INT "${NUM_EXPERIMENTS}") # Convert to integer for loop
if(NUM_EXPERIMENTS_AS_INT LESS_EQUAL 0)
    message(FATAL_ERROR "NUM_EXPERIMENTS must be a positive integer. Value: ${NUM_EXPERIMENTS}")
endif()

add_subdirectory(opencv/verticalfusion/vertical_fusion_kernel_instances/mul)
add_subdirectory(opencv/verticalfusion/vertical_fusion_kernel_instances/mul1)
