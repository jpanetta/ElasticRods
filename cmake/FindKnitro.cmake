# - Find Knitro
#  Searches for includes/libraries using environment variable $KNITRO_PATH or $KNITRO_DIR
#  KNITRO_INCLUDE_DIRS - where to find knitro.h and (separately) the c++ interface
#  KNITRO_LIBRARIES    - List of libraries needed to use knitro.
#  KNITRO_FOUND        - True if knitro found.


IF (KNITRO_INCLUDE_DIRS)
  # Already in cache, be silent
  SET (knitro_FIND_QUIETLY TRUE)
ENDIF (KNITRO_INCLUDE_DIRS)

FIND_PATH(KNITRO_INCLUDE_DIR knitro.h
	HINTS
        $ENV{KNITRO_PATH}/include
        $ENV{KNITRO_DIR}/include
)

FIND_LIBRARY (KNITRO_LIBRARY NAMES knitro knitro1031
	HINTS
        $ENV{KNITRO_PATH}/lib
        $ENV{KNITRO_DIR}/lib
)

# handle the QUIETLY and REQUIRED arguments and set KNITRO_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (KNITRO DEFAULT_MSG
  KNITRO_LIBRARY
  KNITRO_INCLUDE_DIR)

IF(KNITRO_FOUND)
    SET (KNITRO_LIBRARIES ${KNITRO_LIBRARY})
    SET (KNITRO_INCLUDE_DIRS "${KNITRO_INCLUDE_DIR}" "${KNITRO_INCLUDE_DIR}/../examples/C++/include")
ELSE (KNITRO_FOUND)
    SET (KNITRO_LIBRARIES)
ENDIF (KNITRO_FOUND)

MARK_AS_ADVANCED (KNITRO_LIBRARY KNITRO_INCLUDE_DIR KNITRO_INCLUDE_DIRS KNITRO_LIBRARIES)
