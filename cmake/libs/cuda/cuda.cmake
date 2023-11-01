include(cmake/libs/cuda/archs.cmake)
include(cmake/libs/cuda/debug.cmake)
include(cmake/libs/cuda/target_generation.cmake)
include(cmake/libs/cuda/deploy.cmake)

find_package(CUDAToolkit ${CUDA_NVCC_VERSION_FROM_VERSION_FILE} EXACT REQUIRED)
# some external lbis(opencv) use findCuda, so we set this variable for compatibility
set(CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDAToolkit_LIBRARY_ROOT})

string(REPLACE "\\" "/" CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
 

function(add_cuda_to_target TARGET_NAME COMPONENTS)
    set_default_cuda_target_properties(${TARGET_NAME})
    # we need to deploy runtime because we se CUDA_RUNTIME_LIBRARY property to Shared
    list(APPEND COMPONENTS "cudart")
    add_cuda_debug_support_to_target(${TARGET_NAME})
    
    set(EXPORTED_CUDA_TARGETS ${COMPONENTS})
    set(COMPONENTS_TO_DEPLOY ${COMPONENTS})
    list(TRANSFORM EXPORTED_CUDA_TARGETS PREPEND "CUDA::")
    target_link_libraries(${TARGET_NAME} PRIVATE ${EXPORTED_CUDA_TARGETS})

    # runtime deployment#######
    list(REMOVE_ITEM COMPONENTS_TO_DEPLOY "cuda_driver") # comes with nvidia driver installer
    
    if(NOT UNIX)
        deploy_cuda_dependencies(${TARGET_NAME} "${COMPONENTS_TO_DEPLOY}")
    endif()
endfunction()
