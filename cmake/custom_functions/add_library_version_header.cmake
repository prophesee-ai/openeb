# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(GIT_BRANCH "main")
set(GIT_COMMIT_ID "0abe190e9d21f209cb0e5968ee4ad21d0229e5d1")
set(GIT_COMMIT_DATE "2025-04-24 12:21:13 +0200")

find_program(GIT_SCM git DOC "Git version control" HINTS "C:\\Program Files\\Git\\bin\\")

# If git information are not provided in command line when running cmake, try to automatically determine them
if(NOT GIT_BRANCH)
    set(GIT_COMMAND_GET_BRANCH "\"${GIT_SCM}\" -C \"${PROJECT_SOURCE_DIR}\" rev-parse --abbrev-ref HEAD")
else()
    set(GIT_COMMAND_GET_BRANCH "\"${CMAKE_COMMAND}\" -E echo ${GIT_BRANCH}")
endif(NOT GIT_BRANCH)

if(NOT GIT_COMMIT_ID)
    set(GIT_COMMAND_GET_COMMIT_ID "\"${GIT_SCM}\" -C \"${PROJECT_SOURCE_DIR}\" log -1 --pretty=format:%H")
else()
    set(GIT_COMMAND_GET_COMMIT_ID "\"${CMAKE_COMMAND}\" -E echo ${GIT_COMMIT_ID}")
endif(NOT GIT_COMMIT_ID)

if(NOT GIT_COMMIT_DATE)
    set(GIT_COMMAND_GET_COMMIT_DATE "\"${GIT_SCM}\" -C \"${PROJECT_SOURCE_DIR}\" log -1 --pretty=format:%cd --date=iso8601")
else()
    set(GIT_COMMAND_GET_COMMIT_DATE "\"${CMAKE_COMMAND}\" -E echo ${GIT_COMMIT_DATE}")
endif(NOT GIT_COMMIT_DATE)

foreach(cmd GIT_COMMAND_GET_BRANCH GIT_COMMAND_GET_COMMIT_ID GIT_COMMAND_GET_COMMIT_DATE)
    string(REPLACE "\"" "\\\"" ${cmd}_QUOTES_ESCAPED "${${cmd}}")
endforeach(cmd)

# Write the cmake file that we'll run to generate the version file
# Remark : we need to do this instead of just having the file and passing the
# commands like ${GIT_COMMAND_GET_BRANCH} into a variable because this variable would
# contain spaces (and even if passing it between quotes id does not work because it keeps the quotes)
set(cmake_script ${GENERATE_FILES_DIRECTORY}/scripts/configure_version_file.cmake CACHE PATH "Path to the cmake script to generate version files")
file(WRITE ${cmake_script} "
include(CMakeParseArguments)
function(wrap_command)
    cmake_parse_arguments(ARGS \"\" \"OUTPUT\" \"COMMAND\" \${ARGN})
    # if cmake is executed as sudo, run the command as the underlying user
    if (UNIX AND (NOT \"\$ENV{SUDO_USER}\" STREQUAL \"\"))
        string (REPLACE \";\" \" \" _cmd \"\${ARGS_COMMAND}\")
        set (\${ARGS_OUTPUT} su \$ENV{SUDO_USER} -c \"\${_cmd}\" PARENT_SCOPE)
    else ()
        set (\${ARGS_OUTPUT} \"\${ARGS_COMMAND}\" PARENT_SCOPE)
    endif ()
endfunction()
wrap_command(COMMAND ${GIT_COMMAND_GET_BRANCH} OUTPUT cmd)
execute_process(
    COMMAND \${cmd}
    OUTPUT_VARIABLE GIT_BRANCH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE err
    RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
    message(FATAL_ERROR \"Error execuding command \n'\${cmd}' :\n\${err}\")
endif()
wrap_command(COMMAND ${GIT_COMMAND_GET_COMMIT_ID} OUTPUT cmd)
execute_process(
    COMMAND \${cmd}
    OUTPUT_VARIABLE GIT_HASH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE err
    RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
    message(FATAL_ERROR \"Error execuding command \n'\${cmd}' :\n\${err}\")
endif()
wrap_command(COMMAND ${GIT_COMMAND_GET_COMMIT_DATE} OUTPUT cmd)
execute_process(
    COMMAND \${cmd}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE err
    RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
    message(FATAL_ERROR \"Error execuding command \n'\${cmd}' :\n\${err}\")
endif()

configure_file(\"${CMAKE_CURRENT_LIST_DIR}/version.h.in\" \"\${OUTPUTFILE}\" @ONLY)
")

########################################################################################
#
# Creates a custom target that writes a version header for a given library
#
#
# usage :
#     add_library_version_header(<target-name> <output-file-path> <library-name>
#        [VERSION X.Y.Z]
#      )
#
#
#  Adds a custom target named <target-name> that writes a version file at <output-file-path>. The header guard of the
#  generated header will be <LIBRARY_NAME_UPPER>_VERSION_H, and the following variables are defined :
#
#        <LIBRARY_NAME_UPPER>_VERSION_MAJOR
#        <LIBRARY_NAME_UPPER>_VERSION_MINOR
#        <LIBRARY_NAME_UPPER>_VERSION_PATCH
#
#        <LIBRARY_NAME_UPPER>_GIT_BRANCH_RAW
#        <LIBRARY_NAME_UPPER>_GIT_HASH_RAW
#        <LIBRARY_NAME_UPPER>_GIT_COMMIT_DATE
#
#  where <LIBRARY_NAME_UPPER> = <library-name> upper case.
#
#  If option VERSION is given, its value will be used to set variables <LIBRARY_NAME_UPPER>_VERSION_{MAJOR,MINOR,PATCH},
#  otherwise the PROJECT_VERSION will be used
#
include(CMakeParseArguments)
function(add_library_version_header target_name outputfile libname)

    set(LIBRARY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(LIBRARY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(LIBRARY_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(LIBRARY_VERSION_SUFFIX ${PROJECT_VERSION_SUFFIX})

    cmake_parse_arguments(LIB_HEADER "" "VERSION" "" ${ARGN})
    if (LIB_HEADER_VERSION)
        string(REPLACE "." ";" LIB_HEADER_VERSION_LIST ${LIB_HEADER_VERSION})
        list(GET LIB_HEADER_VERSION_LIST 0 LIBRARY_VERSION_MAJOR)
        list(GET LIB_HEADER_VERSION_LIST 1 LIBRARY_VERSION_MINOR)
        list(GET LIB_HEADER_VERSION_LIST 2 LIBRARY_VERSION_PATCH)
    endif(LIB_HEADER_VERSION)

    string(TOUPPER "${libname}" LIBRARY_NAME_UPPER)

    set(version_file_command
        ${CMAKE_COMMAND}
        -DOUTPUTFILE=${outputfile}
        -DLIBRARY_NAME_UPPER=${LIBRARY_NAME_UPPER}
        -DLIBRARY_VERSION_MAJOR=${LIBRARY_VERSION_MAJOR}
        -DLIBRARY_VERSION_MINOR=${LIBRARY_VERSION_MINOR}
        -DLIBRARY_VERSION_PATCH=${LIBRARY_VERSION_PATCH}
        -DLIBRARY_VERSION_SUFFIX=${LIBRARY_VERSION_SUFFIX}
        -P ${cmake_script})

    add_custom_target(
        ${target_name} ALL
        COMMAND ${version_file_command}
        COMMENT "Generating version file for library ${libname}"
        VERBATIM
    )
    if (NOT EXISTS ${outputfile})
        # Make sure version.h exist at configure then it's updated at runtime
        execute_process(
            COMMAND ${version_file_command}
            ERROR_VARIABLE err
            RESULT_VARIABLE ret
        )
        if(ret AND NOT ret EQUAL 0)
            message(FATAL_ERROR "Error execuding command \n'${version_file_command}' :\n${err}")
        endif()
    endif()
endfunction(add_library_version_header)
