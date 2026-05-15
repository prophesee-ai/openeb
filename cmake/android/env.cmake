# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if (NOT DEFINED METAVISION_ANDROID_ENV_INCLUDED)
    # Android SDK, NDK and libraries
    if(DEFINED ENV{ANDROID_SDK_DIR})
        set(ANDROID_SDK_DIR $ENV{ANDROID_SDK_DIR})
    else(DEFINED ENV{ANDROID_SDK_DIR})
        get_filename_component(ANDROID_SDK_DIR_EXPECTED_PATH "${ANDROID_NDK}/../.." ABSOLUTE)
        if (EXISTS ${ANDROID_SDK_DIR_EXPECTED_PATH})
            set(ANDROID_SDK_DIR ${ANDROID_SDK_DIR_EXPECTED_PATH})
        endif (EXISTS ${ANDROID_SDK_DIR_EXPECTED_PATH})
    endif(DEFINED ENV{ANDROID_SDK_DIR})
    set(ANDROID_SDK_DIR ${ANDROID_SDK_DIR} CACHE PATH "Path to Android SDK" FORCE)
    if (ANDROID_SDK_DIR)
        message(STATUS "Android SDK found : ${ANDROID_SDK_DIR}")
    else (ANDROID_SDK_DIR)
        message(FATAL_ERROR "Could not find android SDK, make sure it is installed or set the ANDROID_SDK_DIR env. variable to point to it")
    endif (ANDROID_SDK_DIR)

    if(DEFINED ENV{ANDROID_NDK_DIR})
        set(ANDROID_NDK_DIR $ENV{ANDROID_NDK_DIR})
    else(DEFINED ENV{ANDROID_NDK_DIR})
        set(ANDROID_NDK_DIR ${ANDROID_NDK})
    endif(DEFINED ENV{ANDROID_NDK_DIR})
    set(ANDROID_NDK_DIR ${ANDROID_NDK_DIR} CACHE PATH "Path to Android NDK" FORCE)
    if (ANDROID_NDK_DIR)
        message(STATUS "Android NDK found : ${ANDROID_NDK_DIR}")
    else (ANDROID_NDK_DIR)
        message(FATAL_ERROR "Could not find android NDK, make sure it is installed or set the ANDROID_NDK_DIR env. variable to point to it")
    endif (ANDROID_NDK_DIR)

    set(ANDROID_NDK_VERSION ${ANDROID_NDK_REVISION} CACHE INTERNAL "Android NDK version")

    if(DEFINED ENV{ANDROID_ADB})
        set(ANDROID_ADB $ENV{ANDROID_ADB})
    else(DEFINED ENV{ANDROID_ADB})
        set(ANDROID_ADB ${ANDROID_SDK_DIR}/platform-tools/adb)
    endif(DEFINED ENV{ANDROID_ADB})
    set(ANDROID_ADB ${ANDROID_ADB} CACHE PATH "Path to Android adb tool" FORCE)
    if (ANDROID_ADB)
        message(STATUS "ADB found : ${ANDROID_ADB}")
    else (ANDROID_NDK_DIR)
        message(FATAL_ERROR "Could not find adb, make sure it is installed or set the ANDROID_ADB env. variable to point to it")
    endif (ANDROID_ADB)

    message(STATUS "Android ABI : ${ANDROID_ABI}")
    message(STATUS "Android NDK : ${ANDROID_NDK_VERSION}")

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs/${ANDROID_ABI} CACHE PATH "Output directory of libraries." FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs/${ANDROID_ABI} CACHE PATH "Output directory of archives." FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bins/${ANDROID_ABI} CACHE PATH "Output directory of runtimes." FORCE)
endif (NOT DEFINED METAVISION_ANDROID_ENV_INCLUDED)

if (NOT DEFINED ANDROID_PREBUILT_3RDPARTY_DIR)
  # if not defined, we are using this in config mode from the prebuilt archive
  set(ANDROID_PREBUILT_3RDPARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../../)
endif (NOT DEFINED ANDROID_PREBUILT_3RDPARTY_DIR)

# libusb has no config module, we need to create imported targets by hand
set(_libusb_root ${ANDROID_PREBUILT_3RDPARTY_DIR}/libusb-1.0.22)
set(LIBUSB_INCLUDE_DIR ${_libusb_root}/include/libusb1.0)
set(LIBUSB_LIBRARY ${_libusb_root}/libs/${ANDROID_ABI}/libusb1.0.so)
if (NOT TARGET libusb-1.0)
  add_library(libusb-1.0 SHARED IMPORTED)
  set_target_properties(libusb-1.0 PROPERTIES IMPORTED_LOCATION ${_libusb_root}/libs/${ANDROID_ABI}/libusb1.0.so)
  set_target_properties(libusb-1.0 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${_libusb_root}/include/libusb-1.0)
  # Implicit dependencies
  set_target_properties(libusb-1.0 PROPERTIES INTERFACE_LINK_LIBRARIES log)
endif (NOT TARGET libusb-1.0)

# Boost has a config module
set(Boost_USE_STATIC_LIBS ON)
list(APPEND CMAKE_FIND_ROOT_PATH ${ANDROID_PREBUILT_3RDPARTY_DIR}/boost-1.79.0/libs/arm64-v8a/cmake)
set(Boost_DIR ${ANDROID_PREBUILT_3RDPARTY_DIR}/boost-1.79.0/libs/arm64-v8a/cmake CACHE PATH "Path to precompiled android boost")

# OpenCV comes with its own module, let's use it
set(OpenCV_DIR ${ANDROID_PREBUILT_3RDPARTY_DIR}/opencv-4.8.0/sdk/native/jni CACHE PATH "Path to precompiled android opencv")

# GTest and GMock
set(GTest_DIR "${ANDROID_PREBUILT_3RDPARTY_DIR}/googletest-1.10.0/lib/cmake/GTest")
find_package(GTest CONFIG)

# Eigen3 comes with its own module, let's use it
set(Eigen3_DIR ${ANDROID_PREBUILT_3RDPARTY_DIR}/eigen3-3.3.90/share/eigen3/cmake)

set(METAVISION_ANDROID_ENV_INCLUDED
  TRUE
  CACHE
  INTERNAL "Prevents multiple definition of Android related variables"
)
