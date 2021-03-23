# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
pybind11_create_module(sdk_core module_name)

pybind11_target_sources(${module_name}_python3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colors_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_x_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_y_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/on_demand_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mostrecent_timestamp_buffer_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/periodic_frame_generation_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polarity_filter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polarity_inverter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roi_filter_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_cd_events_buffer_producer_wrapper_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timesurface_producer_algorithm_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_core_bindings.cpp
)

pybind11_target_link_libraries(${module_name}_python3
    PRIVATE
        MetavisionSDK::core
        MetavisionUtils::pybind
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    pybind11_add_dependencies(${module_name}_python3 metavision_sdk_core_python_doc_from_cpp)
    pybind11_target_include_directories(${module_name}_python3 PRIVATE ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    pybind11_target_compile_definitions(${module_name}_python3 PRIVATE GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()
