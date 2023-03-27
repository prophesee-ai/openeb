# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(metavision_sdk_driver_tests_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/biases_gtest.cpp
)
if (HDF5_FOUND)
    list(APPEND metavision_sdk_driver_tests_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/event_file_reader_writer_gtest.cpp
    )
endif ()

add_executable(gtest_metavision_sdk_driver ${metavision_sdk_driver_tests_srcs})
if (HDF5_FOUND)
    target_compile_definitions(gtest_metavision_sdk_driver
        PRIVATE
            HAS_HDF5
            ${HDF5_DEFINITIONS}
            ${HDF5_CXX_DEFINITIONS}
    )
    target_include_directories(gtest_metavision_sdk_driver
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(gtest_metavision_sdk_driver 
        PRIVATE
            ${HDF5_LIBRARIES} 
            hdf5_ecf_codec
    )
endif ()
target_link_libraries(gtest_metavision_sdk_driver
    PRIVATE
        MetavisionSDK::driver
        MetavisionUtils::gtest-main
        Boost::filesystem
)

if (TARGET metavision_hal_gtest_utils)
    target_sources(gtest_metavision_sdk_driver PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_generation_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_stage_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_gtest.cpp
    )
    target_link_libraries(gtest_metavision_sdk_driver PRIVATE metavision_hal_gtest_utils)
endif (TARGET metavision_hal_gtest_utils)

target_include_directories(gtest_metavision_sdk_driver PRIVATE 
    ${CMAKE_SOURCE_DIR}/sdk/modules/driver/cpp/src/include
    ${GENERATE_FILES_DIRECTORY}
)

register_gtest(TEST sdk-driver-unit-tests TARGET gtest_metavision_sdk_driver)