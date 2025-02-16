
set (LAUNCH_SOURCES ${CMAKE_SOURCE_DIR}/tests/main.cpp ${CMAKE_SOURCE_DIR}/tests/main.h)

function (enable_intellisense TARGET_NAME)
	# Hack to get intellisense working for CUDA includes
	target_link_libraries(${TARGET_NAME} PRIVATE cvGPUSpeedup)        
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
		string(FIND ${cuda_source} "fused_kernel_library" is_fkl)
    	string(FIND ${cuda_source} "npp" is_npp)
    	string(FIND ${cuda_source} "opencv" is_ocv)
		string(FIND ${cuda_source} "vertical_fusion_kernel_instances" is_vfki)
	 
		if (${is_vfki} EQUAL -1)
			if (${OPENCV_FOUND})
				if (${is_ocv} GREATER -1)
					add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})				 
					add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs"  )
					target_link_libraries(${cuda_target} PRIVATE vertical_fusion_kernels)
					enable_intellisense(${cuda_target})
				endif()
			endif()
			
			if (${is_npp} GREATER -1)
				add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})			
				target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig) 					
				enable_intellisense(${cuda_target})
			endif()
		endif()
    endforeach()
endfunction()

# Handle vertical_fusion_kernel_instances separately
file(
    GLOB_RECURSE
    VF_KERNEL_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/*.cu"
)

add_library(vertical_fusion_kernels STATIC ${VF_KERNEL_SOURCES})
target_link_libraries(vertical_fusion_kernels PRIVATE cvGPUSpeedup)
set_target_properties(vertical_fusion_kernels PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(vertical_fusion_kernels PRIVATE "${CMAKE_SOURCE_DIR}")
add_opencv_to_target(vertical_fusion_kernels "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs"  )
set_target_cuda_arch_flags(vertical_fusion_kernels)
add_cuda_to_target(vertical_fusion_kernels "")

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