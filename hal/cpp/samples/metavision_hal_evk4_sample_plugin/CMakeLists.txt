# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Add sample plugin object library
add_library(hal_evk4_sample_plugin SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_antiflicker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_camera_synchronization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_camera_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_data_transfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_device_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_digital_crop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_digital_event_mask.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_event_trail_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_erc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_file_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_hw_identification.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_ll_biases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/internal/sample_register_access.cpp
)

target_include_directories(hal_evk4_sample_plugin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hal_evk4_sample_plugin
    PUBLIC
        metavision_hal
        libusb-1.0
)

# Instead of setting the RUNTIME/LIBRARY_OUTPUT_DIRECTORY property on the target, we manually copy
# the library : this will work for linux and windows and avoid the automatic copy of the DLLs the
# plugin depends on by MSVC
add_custom_command(TARGET hal_evk4_sample_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_BUILD_PLUGIN_PATH}/sample_evk4"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hal_evk4_sample_plugin>" "${HAL_BUILD_PLUGIN_PATH}/sample_evk4")

# Install sample
install(DIRECTORY include src
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_evk4_sample_plugin
        COMPONENT metavision-hal-samples
)
install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_evk4_sample_plugin
        COMPONENT metavision-hal-samples
)
