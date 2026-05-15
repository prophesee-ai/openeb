# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_hal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_transfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/demangle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hal_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hal_software_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_raw_data_producer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_file_header.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/resources_folder.cpp
)
