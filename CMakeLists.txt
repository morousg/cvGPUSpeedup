cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 10)
set (PROJECT_VERSION_REV 0)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})
include (cmake/cmake_init.cmake)
include (cmake/targets/virtualfolders.cmake)
cmake_policy(SET CMP0146 OLD) 
cmake_policy(SET CMP0147 OLD) 

include (cmake/deploy/deploy_dependencies.cmake)
include (cmake/libs/cuda/archs.cmake)
include (cmake/generators/version_header.cmake)
project(cvGPUSpeedup VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA 
DESCRIPTION "Make OpenCV-CUDA code run faster on the GPU" HOMEPAGE_URL "https://github.com/morousg/cvGPUSpeedup" )

include (cmake/libs/cuda/cuda.cmake)
include (cmake/cuda_init.cmake)
include (cmake/libs/opencv.cmake)
add_subdirectory(tests)
