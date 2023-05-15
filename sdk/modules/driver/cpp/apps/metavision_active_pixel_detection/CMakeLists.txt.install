# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_active_pixel_detection)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionPSEEHWLayer REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV COMPONENTS highgui REQUIRED)

add_executable(metavision_active_pixel_detection metavision_active_pixel_detection.cpp)
target_link_libraries(metavision_active_pixel_detection PRIVATE Metavision::PSEEHWLayer MetavisionSDK::driver MetavisionSDK::core Boost::program_options ${OpenCV_LIBS})

