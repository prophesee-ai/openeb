# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(sdk_stream_python_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_stream_slicer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_event_file_writer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_stream_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_evt2_event_file_writer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_stream_slicer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_system_builder_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_system_factory_python.cpp
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    set (sdk_stream_python_dependencies metavision_sdk_stream_python_doc_from_cpp)
    set (sdk_stream_python_include_directories ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    set (sdk_stream_python_compile_definitions GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()

add_sdk_python_bindings(stream
    SOURCES ${sdk_stream_python_srcs}
    LINK_LIBRARIES
        PRIVATE
            MetavisionSDK::stream
            MetavisionUtils::pybind
    INCLUDE_DIRECTORIES PRIVATE ${sdk_stream_python_include_directories}
    COMPILE_DEFINITIONS PRIVATE ${sdk_stream_python_compile_definitions}
    DEPENDENCIES ${sdk_stream_python_dependencies}
)
