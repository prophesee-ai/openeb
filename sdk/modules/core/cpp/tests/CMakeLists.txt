# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(metavision_sdk_core_tests_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/async_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_frame_generation_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cd_frame_generator_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/counter_map_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_x_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_y_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_composer_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_composition_stage_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_generation_stage_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generic_producer_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/index_generator_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/on_demand_frame_generation_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/periodic_frame_generation_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polarity_filter_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rate_estimator_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ring_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roi_filter_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stage_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stream_logger_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_cd_events_buffer_producer_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/time_surface_producer_algorithm_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timing_profiler_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/threaded_process_gtest.cpp
)

add_executable(gtest_metavision_sdk_core ${metavision_sdk_core_tests_srcs})
target_link_libraries(gtest_metavision_sdk_core
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::core
        MetavisionUtils::gtest
)

register_gtest(TEST sdk-core-unit-tests TARGET gtest_metavision_sdk_core)


add_executable(deprecation_warning_sample EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/deprecation_warning_sample.cpp
)
if(NOT WIN32)
    target_compile_options(deprecation_warning_sample PUBLIC "-Werror")
else()
    target_compile_options(deprecation_warning_sample PUBLIC "/WX")
endif()