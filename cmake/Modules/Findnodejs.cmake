# Findnodejs
# --------
#
# Find nodejs
#
# This module finds an installed nodejs.  It sets the following variables:
#
#   NODEJS_FOUND - Set to true if nodejs is found
#   NODEJS_EXECUTABLE - The path to the nodejs executable
#   NODEJS_VERSION  - The version number of the nodejs executable

find_program(NODEJS_EXECUTABLE
    NAMES node nodejs 
    HINTS /usr
          /usr/local
          /opt
          /opt/local
    PATH_SUFFIXES bin)

# If nodejs was found, fill in the rest
if (NODEJS_EXECUTABLE)
    # Set the VERSION
    execute_process(COMMAND ${NODEJS_EXECUTABLE} --version
        OUTPUT_VARIABLE NODEJS_VERSION
        ERROR_VARIABLE NODEJS_version_error
        RESULT_VARIABLE NODEJS_version_result_code)

    if(NODEJS_version_result_code)
        if(NODEJS_FIND_REQUIRED)
            message(SEND_ERROR "Command \"${NODEJS_EXECUTABLE} --version\" failed with output:\n${NODEJS_version_error}")
        else()
            message(STATUS "Command \"${NODEJS_EXECUTABLE} --version\" failed with output:\n${NODEJS_version_error}")
        endif ()
    endif ()
    unset(NODEJS_version_error)
    unset(NODEJS_version_result_code)

    # Remove and newlines
    string (STRIP ${NODEJS_VERSION} NODEJS_VERSION)
    string (REPLACE "v" "" NODEJS_VERSION "${NODEJS_VERSION}") 

    set (NODEJS_FOUND TRUE)
else()
    # Fail on REQUIRED
    if (nodejs_FIND_REQUIRED)
        message(SEND_ERROR "Failed to find nodejs executable")
    endif()
endif ()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(nodejs
    REQUIRED_VARS NODEJS_EXECUTABLE 
    VERSION_VAR NODEJS_VERSION)

mark_as_advanced(NODEJS_EXECUTABLE NODEJS_VERSION)
