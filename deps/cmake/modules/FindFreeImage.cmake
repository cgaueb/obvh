# Search for the FreeImage include directory
find_path(FREEIMAGE_INCLUDE_DIR FreeImage.h
  HINTS ${FreeImage_DIR} "${CMAKE_SOURCE_DIR}/../deps/FreeImage/Dist/x64" "${CMAKE_SOURCE_DIR}/../deps/FreeImage"
  PATHS ${FreeImage_DIR} "${CMAKE_SOURCE_DIR}/../deps/FreeImage/Dist/x64" "${CMAKE_SOURCE_DIR}/../deps/FreeImage"
  DOC "Where the FreeImage headers can be found"
)

# SDL-2.0 is the name used by FreeBSD ports...
# don't confuse it for the version number.
find_library(FREEIMAGE_LIBRARY
  NAMES freeimage
  HINTS ${FreeImage_DIR} "${CMAKE_SOURCE_DIR}/../deps/FreeImage/Dist/x64" "${CMAKE_SOURCE_DIR}/../deps/FreeImage"
  PATHS ${FreeImage_DIR} "${CMAKE_SOURCE_DIR}/../deps/FreeImage/Dist/x64" "${CMAKE_SOURCE_DIR}/../deps/FreeImage"
  DOC "Where the FreeImage Library can be found"
)
set(FreeImage_DIR ${FREEIMAGE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(FreeImage
                                  FAIL_MESSAGE  ="FreeImage was not found. Please update the CMake entry named 'FreeImage_DIR' to point to the folder that contains the FreeImage installation"
                                  REQUIRED_VARS FreeImage_DIR FREEIMAGE_LIBRARY FREEIMAGE_INCLUDE_DIR
                                  VERSION_VAR )
