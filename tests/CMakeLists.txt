function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.h"
        "${DIR}/*.cu"
        "${DIR}/*.cuh"
        "${DIR}/*.hpp"
    )

    foreach(cuda_source ${CUDA_SOURCES})    
        get_filename_component(cuda_target ${cuda_source} NAME_WE)    
        add_executable(${cuda_target} ${cuda_source})
        cmake_path(SET path2 "${DIR}")
        cmake_path(GET path2 FILENAME DIR_NAME)       
        set_property(TARGET ${cuda_target} PROPERTY FOLDER tests/${DIR_NAME})
     #   set_virtual_folders("${DIR_NAME}")
        add_cuda_to_target(${cuda_target} "")
        set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
        add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudaimgproc;cudawarping"  )
        target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}")        
        target_link_libraries(${cuda_target} PRIVATE cvGPUSpeedup)        
        set_target_cuda_arch_flags(${cuda_target})
    endforeach()
endfunction()


MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})


foreach(DIR ${LIST_DIRS})
    discover_tests(${DIR})    
endforeach()
 
