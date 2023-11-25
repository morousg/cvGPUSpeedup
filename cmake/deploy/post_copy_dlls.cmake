if(${BUILD_TYPE} MATCHES "Debug" AND ${CURRENT_DLL} MATCHES "Debug")
    file(COPY "${FILE}" DESTINATION "${OUT_DIR}")
endif()

if(${BUILD_TYPE} MATCHES "Release" AND ${CURRENT_DLL} MATCHES "Release")
    file(COPY "${FILE}" DESTINATION "${OUT_DIR}")
endif()

if(${BUILD_TYPE} MATCHES "RelWithDebInfo" AND ${CURRENT_DLL} MATCHES "RelWithDebInfo")
    file(COPY "${FILE}" DESTINATION "${OUT_DIR}")
endif()
