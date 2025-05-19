
add_executable(MyVerticalFusionApp main.cu ${GENERATED_CU_FILES_LIST})

# --- Include Directories ---
target_include_directories(MyVerticalFusionApp PRIVATE
    ${GENERATED_DIR}                     # For "mulN.h", "mul_launcher.h"
    ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>
    # Add path to OpenCV includes if not found globally
    # e.g. /usr/local/include/opencv4 or C:/OpenCV/build/include
)

# --- Linking ---
# find_package(OpenCV REQUIRED) # Example: Find OpenCV
# target_link_libraries(MyVerticalFusionApp PRIVATE
#    ${OpenCV_LIBS} # Link OpenCV libraries
#    CUDA::cudart   # CUDA runtime (often handled by LANGUAGES CUDA)
# )
# Ensure your OpenCV and CUDA libraries are correctly linked.
# With `project(... LANGUAGES CUDA)`, CUDA::cudart might be automatic.

# To pass NUM_EXPERIMENTS_AS_INT to your C++ code (e.g., main.cu)
target_compile_definitions(MyVerticalFusionApp PRIVATE CPP_NUM_EXPERIMENTS=${NUM_EXPERIMENTS_AS_INT})
# In C++: int max_exp = CPP_NUM_EXPERIMENTS;

