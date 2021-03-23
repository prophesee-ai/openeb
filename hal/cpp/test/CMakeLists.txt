# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(metavision_hal_tests_src
    ${CMAKE_CURRENT_SOURCE_DIR}/device_discovery_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hw_identification_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_monitoring_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_roi_gtest.cpp
)

add_executable(gtest_metavision_hal ${metavision_hal_tests_src})
target_link_libraries(gtest_metavision_hal
    PRIVATE
        metavision_hal
        metavision_hal_discovery
        MetavisionUtils::gtest-main
)

register_gtest(TEST hal-unit-tests TARGET gtest_metavision_hal)

add_library(hal_dummy_test_plugin
    SHARED
        $<TARGET_OBJECTS:metavision_hal_info_obj>
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy_test_raw_file_facilities_builder.cpp
)
target_link_libraries(hal_dummy_test_plugin
    PRIVATE
        metavision_hal_info_obj
    PUBLIC
        metavision_hal
)
add_custom_command(TARGET hal_dummy_test_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hal_dummy_test_plugin>" "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)
target_compile_definitions(gtest_metavision_hal PRIVATE HAL_DUMMY_TEST_PLUGIN="${CMAKE_CURRENT_BINARY_DIR}/plugins")
