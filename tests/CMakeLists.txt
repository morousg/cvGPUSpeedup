cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(cvGPUSpeedup)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/../include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NVCCFLAGS_ONLY_FK "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75")
set(NVCCFLAGS_ONLY_CV "-lopencv_core -lopencv_cudaarithm -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 ${NVCCFLAGS_ONLY_FK} ${NVCCFLAGS_ONLY_CV})

file(GLOB_RECURSE CUDA_SOURCES *.cu)

foreach(file ${CUDA_SOURCES})
	get_filename_component(name ${file} NAME_WE)
	cuda_add_executable(${name} ${file})
	target_link_libraries(${name} ${OpenCV_LIBS})
	target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75>)
endforeach()