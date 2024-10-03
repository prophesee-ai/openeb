# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Add sample plugin object library
add_library(hal_toy_sample_plugin_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_camera_synchronization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_camera_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_data_transfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_device_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_file_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_hw_identification.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_plugin.cpp
)

# The following line is needed because when linking a shared library to an object one,
# the object library needs to be compiled with -fPIC
# cf https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
set_target_properties(hal_toy_sample_plugin_obj
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.13.3" AND NOT APPLE AND NOT MSVC)
    target_link_options(hal_toy_sample_plugin_obj PRIVATE "LINKER:-z,defs")
endif()

target_include_directories(hal_toy_sample_plugin_obj
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hal_toy_sample_plugin_obj
    PUBLIC
        metavision_hal
)

# Add sample plugin library
add_library(hal_toy_sample_plugin SHARED $<TARGET_OBJECTS:hal_toy_sample_plugin_obj>)
target_link_libraries(hal_toy_sample_plugin PRIVATE hal_toy_sample_plugin_obj)
# Instead of setting the RUNTIME/LIBRARY_OUTPUT_DIRECTORY property on the target, we manually copy
# the library : this will work for linux and windows and avoid the automatic copy of the DLLs the
# plugin depends on by MSVC
add_custom_command(TARGET hal_toy_sample_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_BUILD_PLUGIN_PATH}/sample"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hal_toy_sample_plugin>" "${HAL_BUILD_PLUGIN_PATH}/sample")


# Tests
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)

# Install sample
install(DIRECTORY include src
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_toy_sample_plugin
        COMPONENT metavision-hal-samples
)
install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_toy_sample_plugin
        COMPONENT metavision-hal-samples
)

