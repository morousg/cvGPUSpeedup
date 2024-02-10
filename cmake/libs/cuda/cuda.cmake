include(cmake/libs/cuda/archs.cmake)
include(cmake/libs/cuda/debug.cmake)
include(cmake/libs/cuda/target_generation.cmake)
include(cmake/libs/cuda/deploy.cmake)


find_package(CUDAToolkit REQUIRED)
# extra cuda_libraries only detected after project() this is needed for compatibility with old local builds that only
# have cuda in normal location instead of custom location
 
# some external lbis(opencv) use findCuda, so we set this variable for compatibility
set(CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDAToolkit_LIBRARY_ROOT})
# file(TO_CMAKE_PATH $ENV{APIS_PATH_VS2017} APIS_PATH)
string(REPLACE "\\" "/" CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
 

function(get_cuda_component_version COMPONENT COMPONENT_VERSION)
    file(READ ${CUDAToolkit_LIBRARY_ROOT}/version.json CUDA_VERSION_FILE_JSON_STRING)
    string(JSON COMPONENT_JSON_STRING GET ${CUDA_VERSION_FILE_JSON_STRING} ${IDX} ${COMPONENT})
    string(JSON COMPONENT_JSON_STRING_1 GET ${COMPONENT_JSON_STRING} ${IDX} version)
    set(${COMPONENT_VERSION} ${COMPONENT_JSON_STRING_1} PARENT_SCOPE)
endfunction()

# Get the name from the current JSON element.
get_cuda_component_version("cuda" CUDA_VERSION_FROM_VERSION_FILE)
# findcudatookit requires nvcc version instead of cuda sdk version
get_cuda_component_version("cuda_nvcc" CUDA_NVCC_VERSION_FROM_VERSION_FILE)
 
 # split cuda version string
 string(REGEX REPLACE "([0-9]+).[0-9]+.[0-9]+" "\\1" CUDA_VERSION_MAJOR ${CUDA_VERSION_FROM_VERSION_FILE})
 string(REGEX REPLACE "[0-9]+.([0-9]+).[0-9]+" "\\1" CUDA_VERSION_MINOR ${CUDA_VERSION_FROM_VERSION_FILE})
 string(REGEX REPLACE "[0-9]+.[0-9]+.([0-9]+)" "\\1" CUDA_VERSION_REVISION ${CUDA_VERSION_FROM_VERSION_FILE})
 

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
