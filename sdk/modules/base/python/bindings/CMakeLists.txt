# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if (WIN32)
    # dummy module used to check if loading bindings linked with a DLL works
    add_sdk_python_bindings(base_dummy 
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_dummy_bindings.cpp
        LINK_LIBRARIES PRIVATE MetavisionSDK::base
    )

    # module taking care of setting the correct paths for loading bindings with DLL dependencies
    add_sdk_python_bindings(base_paths 
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_paths_bindings.cpp
        LINK_LIBRARIES PRIVATE MetavisionUtils::pybind
    )
endif()

set(sdk_base_python_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/debug_buffer_info_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_cd_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_ext_trigger_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_frame_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generic_header_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/software_info_python.cpp
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    set (sdk_base_python_dependencies metavision_sdk_base_python_doc_from_cpp)
    set (sdk_base_python_include_directories ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    set (sdk_base_python_compile_definitions GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()

add_sdk_python_bindings(base
    SOURCES ${sdk_base_python_srcs}
    LINK_LIBRARIES
        PUBLIC
            MetavisionSDK::base
        PRIVATE
            MetavisionUtils::pybind
    INCLUDE_DIRECTORIES PRIVATE ${sdk_base_python_include_directories}
    COMPILE_DEFINITIONS PRIVATE ${sdk_base_python_compile_definitions}
    DEPENDENCIES ${sdk_base_python_dependencies}
)
