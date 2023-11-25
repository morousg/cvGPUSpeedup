#note: you must build opencv with cuda support (vanilla builds don't provide cuda modules)
#remember to set OpenCV_DIR enviroment varialbe to find opencv or use pkgconfig
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)

function(add_opencv_to_target TARGET_NAME COMPONENTS)
    set(EXPORTED_TARGETS ${COMPONENTS})
    #if(WIN32)
     #   list(TRANSFORM EXPORTED_TARGETS PREPEND "opencv::")
    #elseif(UNIX)
        list(TRANSFORM EXPORTED_TARGETS PREPEND "opencv_")
    #endif()
    # https://stackoverflow.com/questions/55640570/using-opencvs-cmake-module-targets-instead-of-including-all-libraries-directly
    target_link_libraries(${TARGET_NAME} PRIVATE ${EXPORTED_TARGETS})
    if(WIN32)
        deploy_exported_target_dependencies(${TARGET_NAME} ${EXPORTED_TARGETS})
    endif()
    # cuda libraries  implicit dependencies for cuda modules

    if(NOT UNIX)
        deploy_cuda_dependencies(${TARGET_NAME} "${NPP_COMPONENTS}")
        # cudaarithm
        deploy_cuda_dependencies(${TARGET_NAME} cufft)

    endif()

endfunction()
