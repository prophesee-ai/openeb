# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(sdk_ui_python_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/base_window_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_loop_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_ui_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mt_window_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ui_event_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pb_doc_ui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/window_wrapper.h
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    set (sdk_ui_python_dependencies metavision_sdk_ui_python_doc_from_cpp)
    set (sdk_ui_python_include_directories ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    set (sdk_ui_python_compile_definitions GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()

add_sdk_python_bindings(ui
    SOURCES ${sdk_ui_python_srcs}
    LINK_LIBRARIES
        PUBLIC
            MetavisionSDK::ui
        PRIVATE
            MetavisionSDK::core
            MetavisionUtils::pybind
    INCLUDE_DIRECTORIES PRIVATE ${sdk_ui_python_include_directories}
    COMPILE_DEFINITIONS PRIVATE ${sdk_ui_python_compile_definitions}
    DEPENDENCIES ${sdk_ui_python_dependencies}
)
