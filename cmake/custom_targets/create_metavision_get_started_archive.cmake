# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(output_metavision_get_started_archive_dir_path "${GENERATE_FILES_DIRECTORY}/metavision_get_started_archive")

add_custom_target(create_metavision_get_started_archive_folder
    COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DOUTPUT_DIR="${output_metavision_get_started_archive_dir_path}"
        -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
        -DGENERATE_FILES_DIRECTORY="${GENERATE_FILES_DIRECTORY}"
        -P ${CMAKE_CURRENT_LIST_DIR}/create_metavision_get_started_archive_folder.cmake
)

set(output_metavision_get_started_archive_path "${GENERATE_FILES_DIRECTORY}/metavision_get_started_${PROJECT_VERSION_FULL}.tar.gz")

add_custom_target(create_metavision_get_started_archive
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_metavision_get_started_archive_dir_path} ${CMAKE_COMMAND} -E tar czvf ${output_metavision_get_started_archive_path} .
    COMMAND ${CMAKE_COMMAND} -E echo "File ${output_metavision_get_started_archive_path} generated"
)
add_dependencies(create_metavision_get_started_archive create_metavision_get_started_archive_folder)
