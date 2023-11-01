 
function(set_default_cuda_target_properties TARGET_NAME)
   
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY
                                                                                               Shared)
    set_target_cuda_arch_flags(${TARGET_NAME})
    
    # use less precise but faster cuda math methods
    #target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    # parallel compilation of cuda kernels
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads 0>)

endfunction()
