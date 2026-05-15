# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(GIT_LFS_NOT_AVAILABLE True)

include(CMakeParseArguments)
option(LFS_DOWNLOAD_COMPILATION_RESOURCES "Download LFS resources required for compilation step" ON)
option(LFS_DOWNLOAD_VALIDATION_RESOURCES "Download LFS resources required for validation step" ON)

function(_call_lfs_download file_or_dir_to_download_comma_separated)
    set(retries 3)
    set(success FALSE)
    while (NOT success AND retries GREATER 0)
        execute_process(
            COMMAND git lfs pull --include "${file_or_dir_to_download_comma_separated}"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE OUTPUT
            ERROR_VARIABLE ERROR
            RESULT_VARIABLE RESULT
        )
        if (RESULT AND NOT RESULT EQUAL 0)
            message(WARNING "lfs_download error : ${ERROR} output : ${OUTPUT}")
            math(EXPR retries "${retries} - 1")
            if (retries GREATER 0)
                message(STATUS "Retrying download of LFS files")
            endif ()
        else()
            set(success TRUE)
        endif()
    endwhile()
    if (NOT success)
        message(FATAL_ERROR "lfs_download failed after 3 retries")
    endif()
endfunction ()

function(_lfs_download_at_end_of_configure)
    if (DEFINED LFS_ITEMS_TO_DOWNLOAD)
        string(REPLACE ";" ", " files_or_dirs_to_download_comma_separated "${LFS_ITEMS_TO_DOWNLOAD}")
        message(STATUS "Downloading ${files_or_dirs_to_download_comma_separated} with LFS")
        _call_lfs_download("${files_or_dirs_to_download_comma_separated}")
    endif()
endfunction()

# Function to download files or directories with git lfs
#
# Usage:
#   lfs_download([COMPILATION] [VALIDATION] [IMMEDIATE] file_path|dir_path [file_path|dir_path])
#     COMPILATION (resp. VALIDATION) can be passed to indicate that the corresponding resources should
#     only be downloaded if the variable LFS_DOWNLOAD_COMPILATION_RESOURCES (resp. LFS_DOWNLOAD_VALIDATION_RESOURCES)
#     is set (by default, both are set to TRUE, but if compilation and validation steps are split, one of the variables
#     should be unset to optimize the download time)
#
#     IMMEDIATE indicate that the download must be done while the function is called (the default is to
#     accumulate the list of files to download, and perform the actual fetch/checkout at the end of the
#     configure time of the current project)
function(lfs_download)
    cmake_parse_arguments(LFS_DOWNLOAD_ARGS "VALIDATION;COMPILATION;IMMEDIATE" "" "" ${ARGN})

    if (GIT_LFS_NOT_AVAILABLE)
        return()
    endif (GIT_LFS_NOT_AVAILABLE)

    if (NOT LFS_DOWNLOAD_ARGS_IMMEDIATE)
        # Check if we already scheduled a deferred call for LFS to download items
        set(_call_already_scheduled FALSE)
        cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} GET_CALL_IDS _deferred_call_ids)
        foreach(id ${_deferred_call_ids})
            cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} GET_CALL ${id} _deferred_call)
            if("${_deferred_call}" MATCHES "_lfs_download_at_end_of_configure")
                set(_call_already_scheduled TRUE)
                break()
            endif()
        endforeach()

        # Schedule the deferred call if it hasn't been scheduled yet
        if(NOT _call_already_scheduled)
            set(LFS_ITEMS_TO_DOWNLOAD "" CACHE INTERNAL "List of items to download with git LFS")
            cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} CALL _lfs_download_at_end_of_configure())
        endif()
    endif()

    if (NOT LFS_DOWNLOAD_ARGS_VALIDATION AND NOT LFS_DOWNLOAD_ARGS_COMPILATION)
        message(FATAL_ERROR "lfs_download must be called with at least one keyword in [COMPILATION,VALIDATION]")
    endif ()
    if (NOT LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "lfs_download must be called with a list of files/dirs")
    endif ()

    foreach(file_or_dir_to_download ${LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS})
        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download}")
            message(FATAL_ERROR "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download} does not exist")
        endif()
    endforeach(file_or_dir_to_download)
    if (LFS_DOWNLOAD_ARGS_IMMEDIATE)
        string(REPLACE ";" ", " files_or_dirs_to_download_comma_separated "${LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS}")
        message(STATUS "Downloading ${files_or_dirs_to_download_comma_separated} with LFS")
        _call_lfs_download("${files_or_dirs_to_download_comma_separated}")
    else()
        if (DEFINED LFS_ITEMS_TO_DOWNLOAD)
            set (_lfs_items_to_download "${LFS_ITEMS_TO_DOWNLOAD}")
        else ()
            set(_lfs_items_to_download "")
        endif ()
        if ((LFS_DOWNLOAD_ARGS_COMPILATION AND LFS_DOWNLOAD_COMPILATION_RESOURCES) OR (LFS_DOWNLOAD_ARGS_VALIDATION AND LFS_DOWNLOAD_VALIDATION_RESOURCES))
            foreach(file_or_dir_to_download ${LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS})
                if (NOT "${file_or_dir_to_download}" IN_LIST _lfs_items_to_download)
                    list(APPEND _lfs_items_to_download ${file_or_dir_to_download})
                    message(STATUS "Adding ${file_or_dir_to_download} to the list of LFS files to download")
                endif ()
            endforeach(file_or_dir_to_download)
            set(LFS_ITEMS_TO_DOWNLOAD "${_lfs_items_to_download}" CACHE INTERNAL "List of items to download with git LFS")
        endif ()
    endif()
endfunction()
