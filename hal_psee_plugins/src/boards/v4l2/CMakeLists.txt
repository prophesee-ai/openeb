# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if(NOT HAS_V4L2 OR ANDROID)
    return()
endif()

target_compile_definitions(metavision_hal_psee_plugin_obj PRIVATE HAS_V4L2)

target_sources(metavision_hal_psee_plugin_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera_discovery.cpp
)

target_sources(metavision_psee_hw_layer_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_data_transfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_mmap_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_dmabuf_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_controls.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_board_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_hardware_identification.cpp
)
