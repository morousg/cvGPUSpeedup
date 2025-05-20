function (add_vertical_fusion_single_benchmark TARGET_NAME)
    # --- File Generation Loop ---
  
    add_executable(${TARGET_NAME}  ${TARGET_NAME}.cu)            
    add_cuda_to_target(${TARGET_NAME} "")

    # --- Include Directories ---
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include    
        ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>        
        
    )
    add_fkl_to_target(${TARGET_NAME})                 
    add_opencv_to_target(${TARGET_NAME} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
    enable_intellisense(${TARGET_NAME})
    
endfunction()
 
add_vertical_fusion_single_benchmark("benchmark_vertical_fusionCompareDiv")
add_vertical_fusion_single_benchmark("benchmark_vertical_fusionCompareSub")
add_vertical_fusion_single_benchmark("benchmark_vertical_fusionCompareSum")