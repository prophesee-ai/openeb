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
option(LFS_DOWNLOAD_COMPILATION_RESOURCES "Only download LFS resources required for compilation step" ON)
option(LFS_DOWNLOAD_VALIDATION_RESOURCES "Only download LFS resources required for compilation step" ON)

# Function to download files or directories with git lfs
#
# Usage:
#   lfs_download(file_path|dir_path [file_path|dir_path])
function(lfs_download)
    cmake_parse_arguments(LFS_DOWNLOAD_ARGS "VALIDATION;COMPILATION" "" "" ${ARGN})
    if (NOT LFS_DOWNLOAD_ARGS_VALIDATION AND NOT LFS_DOWNLOAD_ARGS_COMPILATION)
        message(FATAL_ERROR "lfs_download must be called with at least one keyword in [COMPILATION,VALIDATION]")
    endif ()
    if (NOT LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "lfs_download must be called with a list of files/dirs")
    endif ()
    string(REPLACE ";" ", " file_or_dir_to_download_comma_separated "${LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS}")
    if (NOT GIT_LFS_NOT_AVAILABLE)
        foreach(file_or_dir_to_download ${LFS_DOWNLOAD_ARGS_UNPARSED_ARGUMENTS})
            if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download}")
                message(FATAL_ERROR "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download} does not exist")
            endif()
        endforeach(file_or_dir_to_download)

        if ((LFS_DOWNLOAD_ARGS_COMPILATION AND LFS_DOWNLOAD_COMPILATION_RESOURCES) OR (LFS_DOWNLOAD_ARGS_VALIDATION AND LFS_DOWNLOAD_VALIDATION_RESOURCES))
            message(STATUS "Downloading ${file_or_dir_to_download_comma_separated} with lfs")
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
                        message(STATUS "Retrying download of ${file_or_dir_to_download_comma_separated}")
                    endif ()
                else()
                    set(success TRUE)
                endif()
            endwhile()
            if (NOT success)
                message(FATAL_ERROR "lfs_download failed after 3 retries")
            endif()
        endif ()
    endif (NOT GIT_LFS_NOT_AVAILABLE)
endfunction()
