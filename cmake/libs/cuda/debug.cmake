option(CUDA_ENABLE_NVTOOLSEXT "Enable CUDA_ENABLE_NVTOOLSEXT " OFF)
option(CUDA_ENABLE_DEBUG "Generate CUDA debug information for device code and turn off all device code optimizations"
       OFF)

function(add_cuda_debug_support_to_target TARGET_NAME)
    # optional debug code in gpu ()
    if(${CUDA_ENABLE_DEBUG})
        target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
    endif()

    if(${CUDA_ENABLE_NVTOOLSEXT})
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_NVTX)
    endif()

endfunction()
