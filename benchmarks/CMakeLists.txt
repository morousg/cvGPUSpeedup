

include (benchconfig.cmake)


function(add_vertical_fusion_kernels TARGET_NAME FOLDER_FROM)
    file(
        GLOB_RECURSE
        VF_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/${FOLDER_FROM}/*"
    )
    target_sources(${TARGET_NAME} PRIVATE ${VF_KERNEL_SOURCES})
endfunction()

function (discover_tests DIR)
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
    )
    
    foreach(cuda_source ${CUDA_SOURCES})
    get_filename_component(cuda_target ${cuda_source} NAME_WE)
                
        string(FIND ${cuda_source} "opencv" is_ocv)
        string(FIND ${cuda_source} "vertical_fusion_kernel_instances" is_vfki)

        # Detect this 4 particualr cases to compile the kernels they need as a library
        string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMul.cu" is_vmul)
        string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMul1.cu" is_vmul1)
        string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMulAdd.cu" is_vmulsum)
        string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMulAdd1.cu" is_vmulsum1)

        if (${is_vfki} EQUAL -1)
             
                if (${is_ocv} GREATER -1)
                    add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})

                    # If the source is one of the 4 cases, add the required kernels as a library
                    if (${is_vmul} GREATER -1)
                        add_vertical_fusion_kernels(${cuda_target} "mul")
                    elseif (${is_vmul1} GREATER -1)
                        add_vertical_fusion_kernels(${cuda_target} "mul1")
                    elseif (${is_vmulsum} GREATER -1)
                        add_vertical_fusion_kernels(${cuda_target} "mul_add")
                    elseif (${is_vmulsum1} GREATER -1)
                        add_vertical_fusion_kernels(${cuda_target} "mul_add1")
                    endif()
                    add_fkl_to_target(${cuda_target})
                    target_include_directories(${cuda_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)                    
                    add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
                    enable_intellisense(${cuda_target})
                endif()
     
             
        endif()
    endforeach()
endfunction()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
 
foreach(DIR ${LIST_DIRS})
    discover_tests(${DIR})    
endforeach()