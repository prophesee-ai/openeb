# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_hal_treuzell_protocol_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_streamer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_device_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_regmap_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_issd_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_psee_fpga_device.cpp
)

target_sources(metavision_hal_psee_video_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_psee_video.cpp
)
if(TARGET hal_plugin_prophesee)
target_sources(hal_plugin_prophesee PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tz_psee_video_build.cpp
)
endif()

target_sources(metavision_hal_ti_tmp103_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ti_tmp103.cpp
)
