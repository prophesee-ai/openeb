# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Utils library for encoding with PSEE plugin format :
add_library(metavision_hal_psee_plugins_gtest_utils INTERFACE)
target_include_directories(metavision_hal_psee_plugins_gtest_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest_utils
)
target_link_libraries(metavision_hal_psee_plugins_gtest_utils
    INTERFACE
        metavision_hal_psee_plugin_obj
)

# Tests for PSEE plugins
set(metavision_hal_psee_plugins_tests_src
    ${CMAKE_CURRENT_SOURCE_DIR}/device_discovery_psee_plugins_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_encoders_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_data_transfer_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_events_stream_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gen31_event_rate_noise_filter_module_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_events_stream_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/future/i_events_stream_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/psee_decoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tencoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timer_high_encoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_file_header_gtest.cpp
)
add_executable(gtest_metavision_hal_psee_plugins ${metavision_hal_psee_plugins_tests_src} $<TARGET_OBJECTS:metavision_hal_psee_plugin_obj>)
target_link_libraries(gtest_metavision_hal_psee_plugins
    PRIVATE
        metavision_hal
        metavision_hal_discovery
        metavision_hal_psee_plugins_gtest_utils
        MetavisionUtils::gtest-main
        Boost::filesystem
)

target_include_directories(gtest_metavision_hal_psee_plugins
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HAL_GEN4_SUPPORT)
    target_compile_definitions(gtest_metavision_hal_psee_plugins PRIVATE "HAL_GEN4_SUPPORT")
endif(HAL_GEN4_SUPPORT)

register_gtest(TEST hal-psee-plugins-unit-tests TARGET gtest_metavision_hal_psee_plugins)
