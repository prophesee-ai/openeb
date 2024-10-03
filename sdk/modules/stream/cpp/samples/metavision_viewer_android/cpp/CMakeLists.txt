# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_viewer)
cmake_minimum_required(VERSION 2.8)

get_filename_component(METAVISION_BUILD_DIR "${CMAKE_BINARY_DIR}/../../../../../../../../../.." ABSOLUTE)
set(ANDROID_PREBUILT_3RDPARTY_DIR ${METAVISION_BUILD_DIR}/generated/android/3rdparty/prebuilt)

set(MetavisionHAL_DIR ${METAVISION_BUILD_DIR}/generated/share/cmake/MetavisionHALCMakePackagesFilesDir)
set(MetavisionSDK_DIR ${METAVISION_BUILD_DIR}/generated/share/cmake/MetavisionSDKCMakePackagesFilesDir)
find_package(MetavisionSDK COMPONENTS stream REQUIRED)

set (boost_components_to_find program_options timer chrono thread)
find_package(Boost COMPONENTS ${boost_components_to_find} REQUIRED)

find_package(OpenCV COMPONENTS core imgproc REQUIRED)

set(metavision_viewer_src)
add_subdirectory(src)

add_library(metavision_viewer SHARED ${metavision_viewer_src})
target_link_libraries(metavision_viewer MetavisionSDK::stream opencv_core opencv_imgproc Boost::program_options log android jnigraphics)
