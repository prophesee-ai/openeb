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
    pybind11_add_module(metavision_sdk_base_dummy_python3 MODULE)
    pybind11_target_sources(metavision_sdk_base_dummy_python3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_dummy_bindings.cpp)
    pybind11_target_link_libraries(metavision_sdk_base_dummy_python3
        PRIVATE
            MetavisionSDK::base
    )
    pybind11_set_target_properties(metavision_sdk_base_dummy_python3
        PROPERTIES
            OUTPUT_NAME "metavision_sdk_base_dummy"
            LIBRARY_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
    )
    pybind11_install(TARGETS metavision_sdk_base_dummy_python3
            RUNTIME
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT metavision-sdk-base-paths-internal-python-local-install
            LIBRARY
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT metavision-sdk-base-paths-internal-python-local-install
            ARCHIVE
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT metavision-sdk-base-paths-internal-python-local-install
    )

    # module taking care of setting the correct paths for loading bindings with DLL dependencies
    pybind11_create_module(sdk_base_paths module_name_paths)
    pybind11_target_sources(${module_name_paths}_python3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_paths_bindings.cpp)
    pybind11_target_link_libraries(${module_name_paths}_python3
        PRIVATE
            MetavisionUtils::pybind)
endif()

pybind11_create_module(sdk_base module_name)

pybind11_target_sources(${module_name}_python3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/debug_buffer_info_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_cd_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_ext_trigger_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generic_header_python.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metavision_sdk_base_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/software_info_python.cpp)

pybind11_target_link_libraries(${module_name}_python3
    PUBLIC
        MetavisionSDK::base
    PRIVATE
        MetavisionUtils::pybind
)

if (GENERATE_DOC_PYTHON_BINDINGS)
    pybind11_add_dependencies(${module_name}_python3 metavision_sdk_base_python_doc_from_cpp)
    pybind11_target_include_directories(${module_name}_python3 PRIVATE ${GENERATE_PYTHON_BINDINGS_DOC_DIRECTORY})
    pybind11_target_compile_definitions(${module_name}_python3 PRIVATE GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS)
endif()
