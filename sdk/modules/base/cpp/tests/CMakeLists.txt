# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(metavision_sdk_base_tests_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/bit_instructions_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generic_header_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/object_pool_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/software_info_gtest.cpp
)

add_executable(gtest_metavision_sdk_base ${metavision_sdk_base_tests_srcs})
target_link_libraries(gtest_metavision_sdk_base
    PRIVATE
        MetavisionSDK::base
        MetavisionUtils::gtest
        MetavisionUtils::gtest-main
)

register_gtest(TEST sdk-base-unit-tests TARGET gtest_metavision_sdk_base)