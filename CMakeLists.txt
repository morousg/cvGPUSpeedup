cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(cvGPUSpeedup)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NVCCFLAGS_ONLY_FK "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75")
set(NVCCFLAGS_ONLY_CV "-lopencv_core -lopencv_cudaarithm -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 ${NVCCFLAGS_ONLY_FK} ${NVCCFLAGS_ONLY_CV})

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")

foreach(cuda_source ${CUDA_SOURCES})
    get_filename_component(cuda_target ${cuda_source} NAME_WE)
    cuda_add_executable(${cuda_target} ${cuda_source})
	target_link_libraries(${cuda_target} ${OpenCV_LIBS})
    target_include_directories(${cuda_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/tests")
	target_compile_options(${cuda_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75>)
endforeach()