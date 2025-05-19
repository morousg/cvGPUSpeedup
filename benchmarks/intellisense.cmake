
function (enable_intellisense TARGET_NAME)
    # Hack to get intellisense working for CUDA includes    
    set_target_cuda_arch_flags(${TARGET_NAME})
    add_test(NAME  ${TARGET_NAME} COMMAND ${TARGET_NAME})

    cmake_path(SET path2 "${DIR}")
    cmake_path(GET path2 FILENAME DIR_NAME)
    set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER benchmarks/${DIR_NAME})
    add_cuda_to_target(${TARGET_NAME} "")
    
    if(${ENABLE_DEBUG})
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()

    if(${ENABLE_NVTX})
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()

    if(${ENABLE_BENCHMARK})
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARK)
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
  
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")

endfunction()