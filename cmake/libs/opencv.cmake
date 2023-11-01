#note: you must build opencv with cuda support (vanilla builds don't provide cuda modules)
set (OpenCV_DIR  $ENV{APPDATA}/AutomaticTV/lib/opencv-4.8.0.cuda118.cudnn892.delayedDownload)
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

    set(NPP_COMPONENTS
        nppc
        nppial
        nppicc
        nppig
        nppidei
        nppitc
        nppist
        nppc
        nppif
        nppim
        npps)
    if(NOT UNIX)
        deploy_cuda_dependencies(${TARGET_NAME} "${NPP_COMPONENTS}")
        # cudaarithm
        deploy_cuda_dependencies(${TARGET_NAME} cufft)

    endif()

endfunction()
