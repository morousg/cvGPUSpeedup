cmake_minimum_required (VERSION 3.11.0)

####################################################################################################
# macro to parse macro named arguments
####################################################################################################
include(cmake/parsearguments.cmake)
include(cmake/virtualfolders.cmake)

###############################
#DeviceToDevice project
###############################
#name of the project 
project(CUDALearning LANGUAGES CXX CUDA)
#include (cmake/libs/cuda.cmake)



# avoid minRelSize/RelWithDebugInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)

add_subdirectory(CUDAMemPatterns)