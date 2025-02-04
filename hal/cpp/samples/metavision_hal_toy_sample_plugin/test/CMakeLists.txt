# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

lfs_download(datasets/openeb/sample_plugin_recording.raw VALIDATION)

add_executable(gtest_metavision_hal_toy_sample_plugin 
    ${CMAKE_CURRENT_SOURCE_DIR}/hal_toy_sample_plugin_gtest.cpp
    $<TARGET_OBJECTS:hal_toy_sample_plugin_obj>
)
target_link_libraries(gtest_metavision_hal_toy_sample_plugin
    PRIVATE
        metavision_hal
        metavision_hal_discovery
        MetavisionUtils::gtest
        hal_toy_sample_plugin_obj
)

# Gtest
register_gtest(TEST hal-toy-sample-plugin-test-lib
               TARGET gtest_metavision_hal_toy_sample_plugin
               HAL_PLUGIN_PATH "${HAL_BUILD_PLUGIN_PATH}/sample")

set(apps_to_test metavision_platform_info metavision_file_info metavision_hal_raw_cutter metavision_file_to_dat)
# TODO MV-167 : add app metavision_hal_ls in list above
foreach(app_name IN LISTS apps_to_test)
    string(REPLACE "_" "-" test_name "${app_name}")
    set(test_name "${test_name}-with-sample-plugin-test")
    set(pytest_file "${CMAKE_CURRENT_LIST_DIR}/${app_name}_with_sample_plugin_pytest.py")

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${app_name}_with_sample_plugin_pytest.py.in")
        set (pytest_file "${GENERATE_FILES_DIRECTORY}/scripts/${app_name}_with_sample_plugin_pytest.py")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/${app_name}_with_sample_plugin_pytest.py.in"
            "${pytest_file}"
            @ONLY
        )
    endif()

    add_pytest(NAME ${test_name} PATH ${pytest_file} WORKING_DIRECTORY $<TARGET_FILE_DIR:${app_name}>)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT
        PYTHONPATH=${PROJECT_SOURCE_DIR}/utils/python # to be able to use prophesee_utils python module
        MV_HAL_PLUGIN_PATH=${HAL_BUILD_PLUGIN_PATH}/sample
        HAL_VERSION_HEADER_PATH=${VERSION_FILE_OUTPUT_PATH}
    )

endforeach(app_name)
