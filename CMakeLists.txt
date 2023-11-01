cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (PROJECT_VERSION_MAJOR 10)
set (PROJECT_VERSION_MINOR 0)
set (PROJECT_VERSION_REV 0)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})
cmake_policy(SET CMP0146 OLD) 
cmake_policy(SET CMP0147 OLD) 

include (cmake/libs/cuda/archs.cmake)
include (cmake/libs/cuda/cuda.cmake)
include (cmake/cuda_init.cmake)

project(cvGPUSpeedup VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA 
DESCRIPTION "Make OpenCV-CUDA code run faster on the GPU" HOMEPAGE_URL "https://github.com/morousg/cvGPUSpeedup" )

include (cmake/libs/opencv.cmake)

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")

foreach(cuda_source ${CUDA_SOURCES})    
    get_filename_component(cuda_target ${cuda_source} NAME_WE)    
    add_executable(${cuda_target} ${cuda_source})
	add_cuda_to_target(${cuda_target} "cuda")
    set_target_properties(${cuda_target}  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_opencv_to_target(${cuda_target})
    target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/tests")
	set_target_cuda_arch_flags(${cuda_target})
endforeach()