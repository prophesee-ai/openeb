# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_simple_window)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(MetavisionSDK COMPONENTS base ui REQUIRED)
find_package(OpenCV COMPONENTS imgcodecs REQUIRED)
find_package(Threads REQUIRED)

set(FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(constants.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp @ONLY)

set (sample metavision_simple_window)
add_executable(${sample} ${sample}.cpp constants.h ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp)
target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${sample} MetavisionSDK::base MetavisionSDK::ui opencv_imgcodecs Threads::Threads)
