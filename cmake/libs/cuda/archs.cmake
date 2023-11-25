include(CMakeDependentOption)
 
set(CMAKE_CUDA_ARCHITECTURES OFF)

# by default we only build locally for the native host arch to save build times and binaries size CMake customizations
# and function definitions
option(CUDA_ARCH_TURING "Build for Turing GPUs" OFF)
option(CUDA_ARCH_AMPERE "Build for Ampere GPUs" OFF)
option(CUDA_ARCH_LOVELACE "Build for Ada lovelace GPUs" OFF)
option(CUDA_ARCH_HOPPER "Build for Hopper GPUs" OFF)

if(${CMAKE_VERSION} GREATER_EQUAL "3.23.0")
    option(CUDA_ARCH_NATIVE "Build for cuda host architecture only" ON)
    cmake_dependent_option(CUDA_ARCH_NATIVE "Build for cuda host architecture only" ON
                                "NOT CUDA_ARCH_ALL;NOT CUDA_ARCH_TURING; NOT CUDA_ARCH_AMPERE; NOT CUDA_ARCH_LOVELACE; NOT CUDA_ARCH_HOPPER" OFF)


    option(CUDA_ARCH_ALL "Build all cuda architectures " OFF)
    cmake_dependent_option(CUDA_ARCH_ALL "Build all supported cuda archs" OFF
                            "NOT CUDA_ARCH_NATIVE; NOT CUDA_ARCH_TURING; NOT CUDA_ARCH_AMPERE; NOT CUDA_ARCH_LOVELACE; NOT CUDA_ARCH_HOPPER" OFF)

    if(${CUDA_ARCH_NATIVE})
            message(
                STATUS
                    "Building cuda kernels for host arch only."
            )
    endif()
endif()        
# build archs controlled by cmake options must by either native, all OR at least one of these(turing|ampere|ada|hopper|)
function(get_archs_to_build CUDA_SERVER_SUPPORT)
    if(${CMAKE_VERSION} GREATER_EQUAL "3.23.0")
        if(${CUDA_ARCH_NATIVE})
            list(APPEND CUDA_SERVER_SUPPORT1 "native")        
        endif()

        if(${CUDA_ARCH_ALL})
            list(APPEND CUDA_SERVER_SUPPORT1 "all")        
        endif()
    endif()

    if(${CUDA_ARCH_TURING})
        list(APPEND CUDA_SERVER_SUPPORT1 "75-real")
    endif()

    if(${CUDA_ARCH_AMPERE})
        list(APPEND CUDA_SERVER_SUPPORT1 "86-real")
    endif()

    if(${CUDA_ARCH_LOVELACE})
        list(APPEND CUDA_SERVER_SUPPORT1 "89-real")
    endif()

    if(${CUDA_ARCH_HOPPER})
        list(APPEND CUDA_SERVER_SUPPORT1 "90-real")
    endif()    
    set(CUDA_SERVER_SUPPORT ${CUDA_SERVER_SUPPORT1} PARENT_SCOPE)
endfunction()

# cmake 3.28 requests this, but we will override at target level anyway
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native) #
endif()

function(set_target_cuda_arch_flags TARGET_NAME)
    get_archs_to_build(CUDA_SERVER_SUPPORT)    
    if (NOT CUDA_SERVER_SUPPORT)
        message(FATAL_ERROR  "no cuda architecture defined. Aborting project generations" )
    else()
        #message(STATUS "Building ${TARGET_NAME} with these archs:" ${CUDA_SERVER_SUPPORT} )
    endif()
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_SERVER_SUPPORT})
endfunction()

