# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_psee_hw_layer_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_tz_trigger_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_roi_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_roi_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_roi_pixel_mask_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_roi_pixel_reset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_ll_biases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_erc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_nfl_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_nfl_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_dem_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_dem_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_digital_crop.cpp
)

target_sources(metavision_hal_psee_plugin_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/genx320_cx3_tz_device.cpp
)
