cmake_policy(SET CMP0104 NEW) # Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA

 

if(UNIX)
    return()
endif()

 

# for ninja in CI we will always use the NVCC path
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_CUDA_COMPILER ${CUDA_PATH}/nvcc/bin/nvcc.exe)
    set(CUDAToolkit_ROOT ${CUDA_PATH}/nvcc)
endif()
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CUDAToolkit_ROOT ${CUDA_PATH}/nvcc)
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # asume visual studio 2017 or newer other visual studio versions (2019) might be supported in the future
    set(ARCH "x64")
    set(CMAKE_GENERATOR_PLATFORM ${ARCH})
    set(CMAKE_GENERATOR_TOOLSET "host=${ARCH}")
endif()
