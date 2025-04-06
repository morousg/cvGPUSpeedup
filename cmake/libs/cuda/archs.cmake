include(CMakeDependentOption)
 
set(CMAKE_CUDA_ARCHITECTURES OFF)

# if possible, by default we only build locally for the native host arch to save build times and binaries size CMake customizations
# and function definitions


if(${CMAKE_VERSION} GREATER_EQUAL "3.24.0")
    set(CUDA_ARCH "native" CACHE STRING "Cuda architecture to build")
else()
    #default build for all known builds with old cmake (ubuntu 22.04 and jetpack 6.2)
    set(CUDA_ARCH "75;86;87;89;" CACHE STRING "Cuda architecture to build")
endif()
option(CUDA_ARCH "Build for cuda host architecture only" "native")
# build archs controlled by cmake options must by either native, all OR at least one of these(turing|ampere|ada|hopper|)

function(set_target_cuda_arch_flags TARGET_NAME)        
    set_target_properties( ${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")     
     
endfunction()

