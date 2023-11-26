option(ENABLE_NVTX "Enable NVTX utilities" OFF)
option(ENABLE_DEBUG "Generate CUDA debug information for device code and turn off all device code optimizations"
       OFF)

function(add_cuda_debug_support_to_target TARGET_NAME)
    # optional debug code in gpu ()    
    target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)    
endfunction()

function(add_nvtx_support_to_target TARGET_NAME)    
    if (${CMAKE_VERSION} GREATER_EQUAL "3.25.0")
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nvtx3)
    endif()
        
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_NVTX)    
endfunction()
