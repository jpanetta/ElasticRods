# - Find Spectra
#  Searches for includes in the submodule...
#  SPECTRA_INCLUDE_DIRS - where to find spectra.h
#  SPECTRA_FOUND        - True if Spectra was found.


IF (SPECTRA_INCLUDE_DIRS)
  # Already in cache, be silent
  SET (spectra_FIND_QUIETLY TRUE)
ENDIF (SPECTRA_INCLUDE_DIRS)

FIND_PATH(SPECTRA_INCLUDE_DIR Spectra/SymEigsSolver.h
	HINTS
        ${THIRD_PARTY_DIR}/spectra/include
)

# handle the QUIETLY and REQUIRED arguments and set SPECTRA_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (SPECTRA DEFAULT_MSG
    SPECTRA_INCLUDE_DIR)

MARK_AS_ADVANCED (SPECTRA_INCLUDE_DIR)
