
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

function(add_vertical_fusion_kernels FOLDER_FROM)
    file(
        GLOB_RECURSE
        VF_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/${FOLDER_FROM}/*.cu"
    )

    add_library(vertical_fusion_kernels_${FOLDER_FROM} STATIC ${VF_KERNEL_SOURCES})
    target_link_libraries(vertical_fusion_kernels_${FOLDER_FROM} PRIVATE cvGPUSpeedup)
    set_target_properties(vertical_fusion_kernels_${FOLDER_FROM} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
    target_include_directories(vertical_fusion_kernels_${FOLDER_FROM} PRIVATE "${CMAKE_SOURCE_DIR}")
    add_opencv_to_target(vertical_fusion_kernels_${FOLDER_FROM} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
    set_target_cuda_arch_flags(vertical_fusion_kernels_${FOLDER_FROM})
    add_cuda_to_target(vertical_fusion_kernels_${FOLDER_FROM} "")
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

		# Detect this 4 particualr cases to compile the kernels they need as a library
		string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMul.cu" is_vmul)
		string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMul1.cu" is_vmul1)
		string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMulSum.cu" is_vmulsum)
		string(FIND ${cuda_source} "benchmark_vertical_fusionCompareMulSum1.cu" is_vmulsum1)

		# If the source is one of the 4 cases, add the required kernels as a library
		if (${is_vmul} GREATER -1)
			add_vertical_fusion_kernels("mul")
		elseif (${is_vmul1} GREATER -1)
			add_vertical_fusion_kernels("mul1")
		elseif (${is_vmulsum} GREATER -1)
			add_vertical_fusion_kernels("mul_add")
		elseif (${is_vmulsum1} GREATER -1)
			add_vertical_fusion_kernels("mul_add1")
		endif()

		if (${is_vfki} EQUAL -1)
			if (${OPENCV_FOUND})
				if (${is_ocv} GREATER -1)
					add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})
					add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
					# If the source is one of the 4 cases, link with the created library
					if (${is_vmul} GREATER -1)
						target_link_libraries(${cuda_target} PRIVATE vertical_fusion_kernels_mul)
					elseif (${is_vmul1} GREATER -1)
						target_link_libraries(${cuda_target} PRIVATE vertical_fusion_kernels_mul1)
					elseif (${is_vmulsum} GREATER -1)
						target_link_libraries(${cuda_target} PRIVATE vertical_fusion_kernels_mul_add)
					elseif (${is_vmulsum1} GREATER -1)
						target_link_libraries(${cuda_target} PRIVATE vertical_fusion_kernels_mul_add1)
					endif()
					
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