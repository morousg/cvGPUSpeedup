#we store in variable SRCS all .cpp, .h and .rc files
file(GLOB_RECURSE SRCS *.cpp *.h *.cu )

#we set the visual studio virtual folders
set_virtual_folders("${SRCS}")

#defines for C++ compiler
add_definitions(-D_VARIADIC_MAX=10 -DUNICODE)
set(TARGET_NAME CUDAMemPatterns)

#cuda wraper to perform add_executable using nvcc and C++.
add_executable(${TARGET_NAME} WIN32 ${SRCS})

set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

#libraries to link the target. We link also with variables LIBS defined in the root CMakeLists.txt file
target_link_libraries(${TARGET_NAME})
#

