include(CmakeDependentOption)
#you can control which archs you wish to build against. Currenlty turing, ampere, lovelace and hopper are validated
 
# needed by cmake 3.28
set(CMAKE_CUDA_ARCHITECTURES OFF)

# by default we only build locally for the native host arch to save build times and binaries size CMake customizations
# and function definitions

option(CUDA_ARCH_NATIVE "Build for cuda host architecture only" ON)
cmake_dependent_option(CUDA_ARCH_NATIVE "Build for cuda host architecture only" ON
                           "NOT CUDA_ARCH_ALL" OFF)


option(CUDA_ARCH_ALL "Build all cuda architectures " OFF)
cmake_dependent_option(CUDA_ARCH_ALL "Build all supported cuda archs" ON
                           "NOT CUDA_ARCH_NATIVE" OFF)

if(${CUDA_ARCH_NATIVE})
        message(
            STATUS
                "Building cuda kernels for host arch only."
        )
        else()
        message(
            STATUS
                "Building cuda kernels for all supported arch in this cuda version"
        )

endif()

# build archs controlled by cmake options must by either native, all OR at least one of these(turing|ampere|ada|hopper|)
function(set_target_cuda_arch_flags TARGET_NAME)

    if(${CUDA_ARCH_NATIVE})
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES "native")
        return()
    endif()

    if(${CUDA_ARCH_ALL})
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES "all")
    return()
    endif()
endfunction()
