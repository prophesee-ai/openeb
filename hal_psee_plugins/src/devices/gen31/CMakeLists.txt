# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_hal_gen31_camera_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_analog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_device_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_digital.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_event_rate_noise_filter_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ll_biases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_monitoring.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_pattern_generator_checker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_pattern_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_roi_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_sensor_if_ctrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_sensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_system_control.cpp
)

target_sources(metavision_hal_gen31_fx3_system_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk1_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk1_fpga.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_fx3_device_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_fx3_facilities_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_trigger_event.cpp
)

target_sources(metavision_hal_gen31_evk2_system_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk2_regmap_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk2_tz_device.cpp
)

target_sources(metavision_hal_gen31_evk3_system_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_evk3_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_tz_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_trigger_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_ccam5_trigger_out.cpp
)
