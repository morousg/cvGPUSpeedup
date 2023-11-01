# ######################################################################################################################
# virtual folders: Projects and source code is not located in the same subfolder Thus, we need to set virtual folders.
# Those, have the same tree folder structure
# ######################################################################################################################
macro(set_virtual_folders SRCS)
    # add the version resource file to the root folder
    source_group("" FILES ${VERSION_RC})
    foreach(f ${SRCS})
        # Get the relative path of the file
        file(RELATIVE_PATH SRCGR ${CMAKE_CURRENT_SOURCE_DIR} ${f})
        string(FIND "${SRCGR}" "/" POS)
        # Extract the folder if exists, (Remove the filename part)
        string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

        if(${POS} GREATER -1)
            # Source_group expects \\ (double antislash), not / (slash)
            string(REPLACE / \\ SRCGR ${SRCGR})
            source_group("${SRCGR}" FILES ${f})
        else()
            source_group("" FILES ${f})
        endif()
    endforeach()
endmacro()
