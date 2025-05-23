set(FKL_VERSION_MAJOR 0)
set(FKL_VERSION_MINOR 1)
set(FKL_VERSION_RELEASE 8)
set(FKL_VERSION ${FKL_VERSION_MAJOR}.${FKL_VERSION_MINOR}.${FKL_VERSION_RELEASE})

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/fkl/lib/export")

find_package(FKL ${FKL_VERSION} CONFIG REQUIRED)
function(add_fkl_to_target TARGET_NAME)    
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/fkl/include")
    #target_link_libraries(${TARGET_NAME} PRIVATE FKL::FKL)
endfunction()