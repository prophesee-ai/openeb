# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_psee_hw_layer_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_event_rate_noise_filter_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ll_biases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_roi_command.cpp
)

target_sources(metavision_hal_psee_plugin_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_pattern_generator_checker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_pattern_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk2_tz_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_tz_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_trigger_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_trigger_out.cpp
)
