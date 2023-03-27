# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_psee_hw_layer_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_plugin.cpp)

target_sources(hal_plugin_gen31_evk2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_gen31_evk2.cpp)
target_sources(hal_plugin_gen31_evk3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_gen31_evk3.cpp)
target_sources(hal_plugin_gen41_evk2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_gen41_evk2.cpp)
target_sources(hal_plugin_gen41_evk3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_gen41_evk3.cpp)
target_sources(hal_plugin_imx636_evk2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_imx636_evk2.cpp)
target_sources(hal_plugin_imx636_evk3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_imx636_evk3.cpp)
target_sources(hal_plugin_imx636_evk4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_imx636_evk4.cpp)
target_sources(hal_plugin_prophesee PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_universal.cpp)

if(HAL_SENSORLIB_SUPPORT)
    target_sources(hal_plugin_sensorlib_tz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_sensorlib_tz.cpp)
    target_sources(hal_plugin_sensorlib_fx3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psee_sensorlib_fx3.cpp)
endif(HAL_SENSORLIB_SUPPORT)
