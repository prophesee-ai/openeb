# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(metavision_sdk_stream_tests_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_evt2_event_file_writer_gtest.cpp
)
if (HDF5_FOUND)
    list(APPEND metavision_sdk_stream_tests_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/event_file_reader_writer_gtest.cpp
    )
endif ()

add_executable(gtest_metavision_sdk_stream ${metavision_sdk_stream_tests_srcs})
if (HDF5_FOUND)
    target_compile_definitions(gtest_metavision_sdk_stream
        PRIVATE
            HAS_HDF5
            ${HDF5_DEFINITIONS}
            ${HDF5_CXX_DEFINITIONS}
    )
    target_include_directories(gtest_metavision_sdk_stream
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(gtest_metavision_sdk_stream
        PRIVATE
            ${HDF5_LIBRARIES}
            hdf5_ecf_codec
    )
endif ()
target_link_libraries(gtest_metavision_sdk_stream
    PRIVATE
        MetavisionSDK::stream
        MetavisionUtils::gtest-main
)

if (TARGET metavision_hal_gtest_utils)
    target_sources(gtest_metavision_sdk_stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_generation_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_stream_slicer_gtest.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_stream_slicer_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_gtest.cpp
        # TODO : remove this, we should not have to link with this file, it's only used
        # in the camera gtest because we need functions that are not yet part of HAL I_ROI API
        ${PROJECT_SOURCE_DIR}/hal/cpp/test/dummy_test_plugin_facilities_builder.cpp
    )
    target_link_libraries(gtest_metavision_sdk_stream PRIVATE metavision_hal_gtest_utils)
endif (TARGET metavision_hal_gtest_utils)

add_dependencies(gtest_metavision_sdk_stream hal_dummy_test_plugin)

# Until we execute ctest in docker on Windows, the embedded paths to resources used in the code
# can not be absolute. Here the path is relative to the working directory of the test (which is the
# current binary dir).
target_compile_definitions(gtest_metavision_sdk_stream PRIVATE HAL_DUMMY_TEST_PLUGIN="${CMAKE_CURRENT_BINARY_DIR}/../../../../../hal/cpp/test/plugins")

if(USE_PROTOBUF)
    find_package(Protobuf CONFIG QUIET)
    if (NOT Protobuf_FOUND)
        find_package(Protobuf)
    endif(NOT Protobuf_FOUND)

    if (Protobuf_FOUND)
      target_compile_definitions(gtest_metavision_sdk_stream PRIVATE HAS_PROTOBUF)
    endif()
endif(USE_PROTOBUF)

target_include_directories(gtest_metavision_sdk_stream PRIVATE
    ${PROJECT_SOURCE_DIR}/sdk/modules/stream/cpp/src/include
    ${GENERATE_FILES_DIRECTORY}
)

register_gtest(TEST sdk-stream-unit-tests TARGET gtest_metavision_sdk_stream)
