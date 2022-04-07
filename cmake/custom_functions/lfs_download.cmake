# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(GIT_LFS_NOT_AVAILABLE True)

# Function to download files or directories with git lfs
#
# Usage:
#   lfs_download(file_path|dir_path [file_path|dir_path])
function(lfs_download)
    if (${ARGC} GREATER 0)
        string(REPLACE ";" ", " file_or_dir_to_download_comma_separated "${ARGN}")
        if (NOT GIT_LFS_NOT_AVAILABLE)
            foreach(file_or_dir_to_download ${ARGN})
                if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download}")
                    message(FATAL_ERROR "${PROJECT_SOURCE_DIR}/${file_or_dir_to_download} does not exist")
                endif()
            endforeach(file_or_dir_to_download)
            message(STATUS "Downloading ${file_or_dir_to_download_comma_separated} with lfs")
            execute_process(
                COMMAND git lfs pull --include "${file_or_dir_to_download_comma_separated}"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE OUTUT
                ERROR_VARIABLE ERROR
                RESULT_VARIABLE RESULT
            )
            if (RESULT AND NOT RESULT EQUAL 0)
                message(FATAL_ERROR "lfs_download error : ${ERROR} output : ${OUTPUT}")
            endif ()
        endif (NOT GIT_LFS_NOT_AVAILABLE)
    endif ()
endfunction()
