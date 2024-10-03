# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Tests for PSEE plugins
set(metavision_hal_psee_plugins_tests_src
    ${CMAKE_CURRENT_SOURCE_DIR}/device_discovery_psee_plugins_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_encoders_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_raw_data_producer_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_events_stream_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_event_rate_noise_filter_module_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_events_stream_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/psee_raw_file_decoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_file_header_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/register_map_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/devices/gen31/gen31_ll_biases_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/devices/gen41/gen41_ll_biases_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/devices/imx636/imx636_ll_biases_gtest.cpp
)

add_executable(gtest_metavision_hal_psee_plugins ${metavision_hal_psee_plugins_tests_src} $<TARGET_OBJECTS:metavision_hal_psee_plugin_obj>)
target_link_libraries(gtest_metavision_hal_psee_plugins
    PRIVATE
        metavision_hal
        metavision_hal_gtest_utils
        metavision_hal_discovery
        metavision_hal_psee_plugin_obj
        MetavisionUtils::gtest-main
)

# Need to set RPATH to be able to run test either from build binaries or from a .deb install
set_target_properties(gtest_metavision_hal_psee_plugins PROPERTIES "BUILD_RPATH" "${HAL_BUILD_PLUGIN_PATH};${CMAKE_INSTALL_PREFIX}/${HAL_INSTALL_PLUGIN_RELATIVE_PATH};/usr/${HAL_INSTALL_PLUGIN_RELATIVE_PATH}")

target_include_directories(gtest_metavision_hal_psee_plugins
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gtest_utils
)

register_gtest(TEST hal-psee-plugins-unit-tests TARGET gtest_metavision_hal_psee_plugins)


set(metavision_psee_hw_layer_tests_src
    ${CMAKE_CURRENT_SOURCE_DIR}/psee_hw_layer_gtest.cpp
)

add_executable(gtest_metavision_hal_psee_hw_layer ${metavision_psee_hw_layer_tests_src})
target_link_libraries(gtest_metavision_hal_psee_hw_layer
    PRIVATE
        metavision_hal
        metavision_hal_discovery
        Metavision::PSEEHWLayer
        MetavisionUtils::gtest-main
)

# Need to set RPATH to be able to run test either from build binaries or from a .deb install
set_target_properties(gtest_metavision_hal_psee_hw_layer PROPERTIES "BUILD_RPATH" "${HAL_BUILD_PLUGIN_PATH};${CMAKE_INSTALL_PREFIX}/${HAL_INSTALL_PLUGIN_RELATIVE_PATH};/usr/${HAL_INSTALL_PLUGIN_RELATIVE_PATH}")

register_gtest(TEST hal-psee-hw-layer-unit-tests TARGET gtest_metavision_hal_psee_hw_layer)
