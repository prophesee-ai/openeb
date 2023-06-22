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

    # Get Android ndk version
    file(READ ${ANDROID_NDK_DIR}/source.properties _ndk_source_props)
    string(REGEX MATCH "Pkg\\.Revision = ([0-9.]*)" _ ${_ndk_source_props})
    set(ANDROID_NDK_VERSION ${CMAKE_MATCH_1})
    if ("${ANDROID_NDK_VERSION}" STREQUAL "")
        message(FATAL_ERROR "Could not get Android NDK version from ${ANDROID_NDK_DIR}/source.properties file")
    endif ("${ANDROID_NDK_VERSION}" STREQUAL "")

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

    if(ANDROID_NDK_VERSION VERSION_LESS "22")
        # @TODO Remove once we update to NDK >22
        if (ANDROID_ABI STREQUAL arm64-v8a)
            # arm64-v8a linker seems to not be gold, which causes undefined references
            # due to missing transitive dependencies
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold" CACHE STRING "" FORCE)
        endif (ANDROID_ABI STREQUAL arm64-v8a)
    endif()

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs/${ANDROID_ABI} CACHE PATH "Output directory of libraries." FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs/${ANDROID_ABI} CACHE PATH "Output directory of archives." FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bins/${ANDROID_ABI} CACHE PATH "Output directory of runtimes." FORCE)
endif (NOT DEFINED METAVISION_ANDROID_ENV_INCLUDED)

if (NOT DEFINED ANDROID_PREBUILT_3RDPARTY_DIR)
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


# Boost has no config module, we need to create imported targets by hand
set(_boost_root ${ANDROID_PREBUILT_3RDPARTY_DIR}/boost-1.69.0)
set(_boost_components
  atomic
  chrono
  container
  context
  contract
  coroutine
  date_time
  fiber
  filesystem
  graph
  iostreams
  log
  log_setup
  math_c99
  math_c99f
  math_c99l
  math_tr1
  math_tr1f
  math_tr1l
  prg_exec_monitor
  program_options
  random
  regex
  serialization
  stacktrace_basic
  stacktrace_noop
  system
  thread
  timer
  type_erasure
  unit_test_framework
  wave
  wserialization
)
set(Boost_INCLUDE_DIR "${_boost_root}/include")
set(Boost_FOUND TRUE)
foreach(b_comp ${_boost_components})
  string(TOUPPER ${b_comp} b_comp_uc)
  if (NOT TARGET Boost::${b_comp})
    add_library(Boost::${b_comp} SHARED IMPORTED)
    set_target_properties(Boost::${b_comp} PROPERTIES IMPORTED_LOCATION "${_boost_root}/libs/llvm/${ANDROID_ABI}/libboost_${b_comp}.so")
    set_target_properties(Boost::${b_comp} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_boost_root}/include")
    # Implicit dependencies
    if ("${b_comp}" STREQUAL "filesystem")
      set_target_properties(Boost::filesystem PROPERTIES INTERFACE_LINK_LIBRARIES Boost::system)
    elseif ("${b_comp}" STREQUAL "timer")
      set_target_properties(Boost::timer PROPERTIES INTERFACE_LINK_LIBRARIES Boost::chrono)
    endif ()
  endif (NOT TARGET Boost::${b_comp})
  set(Boost_${b_comp_uc}_FOUND TRUE)
endforeach(b_comp)

# FFMpeg has no config module, we need to create imported targets by hand
set(_ffmpeg_root ${ANDROID_PREBUILT_3RDPARTY_DIR}/mobile-ffmpeg-min-gpl-4.3)
set(_ffmpeg_components
  avcodec
  avdevice
  avfilter
  avformat
  avutil
  swresample
  swscale
)
set(FFMPEG_INCLUDE_DIR "${_ffmpeg_root}/include")
set(FFMPEG_FOUND TRUE)
foreach(b_comp ${_ffmpeg_components})
  string(TOUPPER ${b_comp} b_comp_uc)
  if (NOT TARGET FFMPEG::${b_comp})
    add_library(FFMPEG::${b_comp} SHARED IMPORTED)
    set_target_properties(FFMPEG::${b_comp} PROPERTIES IMPORTED_LOCATION "${_ffmpeg_root}/android-${ANDROID_ABI}/ffmpeg/lib/lib${b_comp}.so")
    set_target_properties(FFMPEG::${b_comp} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_ffmpeg_root}/android-${ANDROID_ABI}/ffmpeg/include")
    # Implicit dependencies
    if ("${b_comp}" STREQUAL "avdevice")
      set_target_properties(FFMPEG::avdevice PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::avfilter)
    elseif ("${b_comp}" STREQUAL "avfilter")
      set_target_properties(FFMPEG::avfilter PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::avformat)
    elseif ("${b_comp}" STREQUAL "avformat")
      set_target_properties(FFMPEG::avformat PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::avcodec)
    elseif ("${b_comp}" STREQUAL "avcodec")
      set_target_properties(FFMPEG::avcodec PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::swresample)
    elseif ("${b_comp}" STREQUAL "swresample")
      set_target_properties(FFMPEG::swresample PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::avutil)
    elseif ("${b_comp}" STREQUAL "swscale")
      set_target_properties(FFMPEG::swscale PROPERTIES INTERFACE_LINK_LIBRARIES FFMPEG::avutil)
    endif ()
  endif (NOT TARGET FFMPEG::${b_comp})
  set(FFMPEG_${b_comp_uc}_FOUND TRUE)
endforeach(b_comp)


# OpenCV comes with its own module, let's use it
set(_opencv_root ${ANDROID_PREBUILT_3RDPARTY_DIR}/opencv-4.0.1/sdk/native)
set(OpenCV_DIR ${_opencv_root}/jni)
# .. however the individual targets are static libraries, so we create a fake one that uses
# libopencv_java which is a shared library as we expect
include_directories(${ANDROID_OPENCV_INC_DIR})
if (NOT TARGET OpenCV::java)
  add_library(OpenCV::java SHARED IMPORTED)
  set_target_properties(OpenCV::java PROPERTIES IMPORTED_LOCATION "${_opencv_root}/libs/${ANDROID_ABI}/libopencv_java4.so")
  set_target_properties(OpenCV::java PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_opencv_root}/jni/include")
  set_target_properties(OpenCV::java PROPERTIES INTERFACE_LINK_LIBRARIES "z;log;android;jnigraphics")
endif (NOT TARGET OpenCV::java)
set(OpenCV_LIBS OpenCV::java)

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
