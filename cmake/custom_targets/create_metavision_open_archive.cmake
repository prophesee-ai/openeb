# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(output_metavision_open_archive_dir_path "${GENERATE_FILES_DIRECTORY}/metavision_open_archive")
set(output_metavision_open_full_archive_dir_path "${GENERATE_FILES_DIRECTORY}/metavision_open_full_archive")

add_custom_target(create_metavision_open_archive_folder
    COMMAND ${CMAKE_COMMAND}
        -DHAL_OPEN_PLUGIN_DEVICES="${HAL_PSEE_OPEN_PLUGIN_DEVICES}"
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DOUTPUT_DIR="${output_metavision_open_archive_dir_path}"
        -DGIT_COMMAND_GET_BRANCH="${GIT_COMMAND_GET_BRANCH}"
        -DGIT_COMMAND_GET_COMMIT_ID="${GIT_COMMAND_GET_COMMIT_ID}"
        -DGIT_COMMAND_GET_COMMIT_DATE="${GIT_COMMAND_GET_COMMIT_DATE}"
        -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
        -DGENERATE_FILES_DIRECTORY="${GENERATE_FILES_DIRECTORY}"
        -P ${CMAKE_CURRENT_LIST_DIR}/create_metavision_open_archive_folder.cmake
    COMMAND ${CMAKE_COMMAND}
        -DHAL_OPEN_PLUGIN_DEVICES="${HAL_PSEE_OPEN_PLUGIN_DEVICES}"
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DOUTPUT_DIR="${output_metavision_open_full_archive_dir_path}/openeb-${PROJECT_VERSION_FULL}"
        -DGIT_COMMAND_GET_BRANCH="${GIT_COMMAND_GET_BRANCH}"
        -DGIT_COMMAND_GET_COMMIT_ID="${GIT_COMMAND_GET_COMMIT_ID}"
        -DGIT_COMMAND_GET_COMMIT_DATE="${GIT_COMMAND_GET_COMMIT_DATE}"
        -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
        -DGENERATE_FILES_DIRECTORY="${GENERATE_FILES_DIRECTORY}"
        -DKEEP_GIT_SUBMODULES=TRUE
        -P ${CMAKE_CURRENT_LIST_DIR}/create_metavision_open_archive_folder.cmake
)

set(output_metavision_open_archive_path "${GENERATE_FILES_DIRECTORY}/metavision_open_${PROJECT_VERSION_FULL}.tar.gz")
set(output_metavision_open_full_archive_path "${GENERATE_FILES_DIRECTORY}/metavision_open_full_${PROJECT_VERSION_FULL}.tar.gz")
add_custom_target(create_metavision_open_archive
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_metavision_open_archive_dir_path} ${CMAKE_COMMAND} -E tar czvf ${output_metavision_open_archive_path} .
    COMMAND ${CMAKE_COMMAND} -E echo "File ${output_metavision_open_archive_path} generated"
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_metavision_open_full_archive_dir_path} ${CMAKE_COMMAND} -E tar czvf ${output_metavision_open_full_archive_path} .
    COMMAND ${CMAKE_COMMAND} -E echo "File ${output_metavision_open_full_archive_path} generated"
)
add_dependencies(create_metavision_open_archive create_metavision_open_archive_folder)
