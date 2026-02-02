# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Utils tests library for encoding with PSEE encoding formats :
add_library(metavision_hal_gtest_utils INTERFACE)
target_include_directories(metavision_hal_gtest_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest_utils
)

set(metavision_hal_tests_src
    ${CMAKE_CURRENT_SOURCE_DIR}/device_config_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_discovery_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_digital_crop_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_digital_event_mask_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hw_identification_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_monitoring_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_roi_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tencoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timer_high_encoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_ll_biases_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders_evt21_decoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders_evt3_decoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders_evt4_decoder_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/data_transfer_gtest.cpp
)

add_executable(gtest_metavision_hal ${metavision_hal_tests_src})
target_link_libraries(gtest_metavision_hal
    PRIVATE
        metavision_hal
        metavision_hal_gtest_utils
        metavision_hal_discovery
        MetavisionUtils::gtest-main
)

register_gtest(TEST hal-unit-tests TARGET gtest_metavision_hal)

add_library(hal_dummy_test_plugin
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy_test_plugin_facilities_builder.cpp
)
target_link_libraries(hal_dummy_test_plugin
    PUBLIC
        metavision_hal
)
add_custom_command(TARGET hal_dummy_test_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hal_dummy_test_plugin>" "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)
add_dependencies(gtest_metavision_hal hal_dummy_test_plugin)

# Until we execute ctest in docker on Windows, the embedded paths to resources used in the code
# can not be absolute. Here the path is relative to the working directory of the test (which is the
# current binary dir).
target_compile_definitions(gtest_metavision_hal PRIVATE HAL_DUMMY_TEST_PLUGIN="plugins")
