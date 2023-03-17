
###############################
#DeviceToDevice project
###############################
#name of the project 
file(GLOB_RECURSE SRCS *.cpp *.h *.cu )
include_directories("${CMAKE_SOURCE_DIR}")
add_executable(CUDAMemPatterns WIN32 ${SRCS})
#include (cmake/libs/cuda.cmake)
set_target_properties(CUDAMemPatterns PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
#set common include dirs and link directories


#Organize projects into virtual folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# avoid minRelSize/RelWithDebugInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
    
   