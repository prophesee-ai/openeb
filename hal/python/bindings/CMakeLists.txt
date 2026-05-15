# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(hal_python_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/device_config_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_discovery_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_antiflicker_module_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_camera_synchronization_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_digital_event_mask.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_digital_crop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_erc_module_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_event_cd_decoder_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_event_ext_trigger_decoder_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_event_frame_decoder_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_event_rate_activity_filter_module_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_events_stream_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_events_stream_decoder_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_geometry_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hal_software_info_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hw_identification_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hw_register_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_ll_biases_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_monitoring_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_roi_pixel_mask_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_event_trail_filter_module_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_plugin_software_info_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_roi_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_trigger_in_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i_trigger_out_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hal_software_info_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_hal_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_file_config_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_file_header_python.cpp
)

if(GENERATE_DOC_PYTHON_BINDINGS)
    set(hal_python_dependencies metavision_hal_python_doc_from_cpp)
    set(hal_python_include_directories ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    set(hal_python_compile_definitions GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()

add_hal_python_bindings(
    SOURCES ${hal_python_srcs}
    LINK_LIBRARIES
    PUBLIC
    metavision_hal
    metavision_hal_discovery
    PRIVATE
    MetavisionUtils::pybind
    INCLUDE_DIRECTORIES PRIVATE ${hal_python_include_directories}
    COMPILE_DEFINITIONS PRIVATE ${hal_python_compile_definitions}
    DEPENDENCIES ${hal_python_dependencies}
)
