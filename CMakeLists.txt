cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (PROJECT_VERSION_MAJOR 10)
set (PROJECT_VERSION_MINOR 0)
set (PROJECT_VERSION_REV 0)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})


include (cmake/libs/cuda/archs.cmake)
include (cmake/libs/cuda/cuda.cmake)
include (cmake/libs/cuda/cuda_init.cmake)

project(cvGPUSpeedup VERSION ${PROJECT_VERSION} HOMEPAGE "https://github.com/morousg/cvGPUSpeedup" LANGUAGES CXX CUDA)

#currently tested against 4.8.0 
set (OPENCV_VERSION_MAJOR 4)
set (OPENCV_VERSION_MINOR 8)
set (OPENCV_VERSION_REV 0)
set (OPENCV_VERSION ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_REV})
find_package(OpenCV VERSION ${OPENCV_VERSION} REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")

foreach(cuda_source ${CUDA_SOURCES})
    
    get_filename_component(cuda_target ${cuda_source} NAME_WE)
    add_executable(${cuda_target} ${cuda_source})
	target_link_libraries(${cuda_target} ${OpenCV_LIBS})
    target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/tests")
	set_target_cuda_arch_flags(${cuda_target})
endforeach()