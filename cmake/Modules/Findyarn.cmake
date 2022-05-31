# Findyarn
# --------
#
# Find yarn
#
# This module finds an installed yarn.  It sets the following variables:
#
#   YARN_FOUND - Set to true if yarn is found
#   YARN_EXECUTABLE - The path to the yarn executable
#   YARN_VERSION  - The version number of the yarn executable

find_program(YARN_EXECUTABLE
    NAMES yarn.cmd yarnpkg yarn
    HINTS /usr
          /usr/local
          /opt
          /opt/local
    PATH_SUFFIXES bin)

# If yarn was found, fill in the rest
if (YARN_EXECUTABLE)
    # Set the VERSION
    execute_process(COMMAND ${YARN_EXECUTABLE} -v
        OUTPUT_VARIABLE YARN_VERSION
        ERROR_VARIABLE YARN_version_error
        RESULT_VARIABLE YARN_version_result_code)

    if(YARN_version_result_code)
        if(YARN_FIND_REQUIRED)
            message(SEND_ERROR "Command \"${YARN_EXECUTABLE} -v\" failed with output:\n${YARN_version_error}")
        else()
            message(STATUS "Command \"${YARN_EXECUTABLE} -v\" failed with output:\n${YARN_version_error}")
        endif ()
    endif ()
    unset(YARN_version_error)
    unset(YARN_version_result_code)

    # Remove and newlines
    string (STRIP ${YARN_VERSION} YARN_VERSION)

    set (YARN_FOUND TRUE)
else()
    # Fail on REQUIRED
    if (yarn_FIND_REQUIRED)
        message(SEND_ERROR "Failed to find yarn executable")
    endif()
endif ()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(yarn
    REQUIRED_VARS YARN_EXECUTABLE 
    VERSION_VAR YARN_VERSION)

mark_as_advanced(YARN_EXECUTABLE YARN_VERSION)