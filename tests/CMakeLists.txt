option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)

set (LAUNCH_SOURCES ${CMAKE_SOURCE_DIR}/tests/main.cpp ${CMAKE_SOURCE_DIR}/tests/main.h)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

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
 
