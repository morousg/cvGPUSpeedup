# Script to copy directory to a destination excluding file matching a pattern Params: ${IN_DIR} ${OUT_DIR}
# ${EXCLUSION_PATTERN}

file(COPY ${IN_DIR} DESTINATION ${OUT_DIR} PATTERN ${EXCLUSION_PATTERN} EXCLUDE)
