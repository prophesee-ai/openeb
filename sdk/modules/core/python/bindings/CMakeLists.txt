# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(sdk_core_python_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_rate_events_splitter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/contrast_map_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colors_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_bbox_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_rescaler_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_preprocessor_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_tracked_box_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/events_integration_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_x_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_y_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/on_demand_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mostrecent_timestamp_buffer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/periodic_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polarity_filter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polarity_inverter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_event_frame_converter_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roi_filter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roi_mask_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rolling_buffer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate_events_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_cd_events_buffer_producer_wrapper_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stream_logger_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose_events_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_core_bindings.cpp
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    set (sdk_core_python_dependencies metavision_sdk_core_python_doc_from_cpp)
    set (sdk_core_python_include_directories ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    set (sdk_core_python_compile_definitions GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()

add_sdk_python_bindings(core
    SOURCES ${sdk_core_python_srcs}
    LINK_LIBRARIES
        PRIVATE
            MetavisionSDK::core
            MetavisionUtils::pybind
    INCLUDE_DIRECTORIES PRIVATE ${sdk_core_python_include_directories}
    COMPILE_DEFINITIONS PRIVATE ${sdk_core_python_compile_definitions}
    DEPENDENCIES ${sdk_core_python_dependencies}
)
