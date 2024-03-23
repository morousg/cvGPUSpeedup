option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)

set (LAUNCH_SOURCES ${CMAKE_SOURCE_DIR}/tests/main.cpp ${CMAKE_SOURCE_DIR}/tests/main.h)

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
		if (OPENCV_FOUND OR is_fkl GREATER -1)
			add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})
			if(${ENABLE_BENCHMARK})
				target_compile_definitions(${cuda_target} PRIVATE ENABLE_BENCHMARK)
			endif()
		  
			cmake_path(SET path2 "${DIR}")
			cmake_path(GET path2 FILENAME DIR_NAME)       
			set_property(TARGET ${cuda_target} PROPERTY FOLDER tests/${DIR_NAME})
			add_cuda_to_target(${cuda_target} "")
			
			if(${ENABLE_DEBUG})
				add_debug_support_to_target(${cuda_target})
			endif()

			if(${ENABLE_NVTX})
				add_nvtx_support_to_target(${cuda_target})
			endif()

			set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
			if (is_fkl EQUAL -1)
				add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs"  )
			endif ()
			target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}")        

			# Hack to get intellisense working for CUDA includes
			target_link_libraries(${cuda_target} PRIVATE cvGPUSpeedup)        
			set_target_cuda_arch_flags(${cuda_target})
			add_test(NAME  ${cuda_target} COMMAND ${cuda_target})
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
 
