# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(metavision_hal_evk4_sample_plugin)

set(CMAKE_CXX_STANDARD 17)
if (MSVC)
    set(CMAKE_DEBUG_POSTFIX "_d")
endif(MSVC)

find_package(MetavisionHAL REQUIRED)

find_path(LIBUSB_INCLUDE_DIR
    NAMES libusb.h
    PATH_SUFFIXES libusb-1.0
)

find_library(LIBUSB_LIBRARY
    NAMES libusb-1.0 usb-1.0
)

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

# Manually copy the library in a separate folder: so that we can set environment variable
# MV_HAL_PLUGIN_PATH to this folder to test the plugin
set(HAL_EVK4_SAMPLE_PLUGIN_PATH "${PROJECT_BINARY_DIR}/lib/hal_evk4_sample_plugin")
add_custom_command(TARGET hal_evk4_sample_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_EVK4_SAMPLE_PLUGIN_PATH}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hal_evk4_sample_plugin>" "${HAL_EVK4_SAMPLE_PLUGIN_PATH}")

target_include_directories(hal_evk4_sample_plugin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${LIBUSB_INCLUDE_DIR}
)

target_link_libraries(hal_evk4_sample_plugin
    PUBLIC
        Metavision::HAL
    PRIVATE
        ${LIBUSB_LIBRARY}
)
