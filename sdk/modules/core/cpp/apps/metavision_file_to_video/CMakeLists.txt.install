# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_file_to_video)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS driver core REQUIRED)
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
find_package(OpenCV COMPONENTS core highgui videoio REQUIRED)
find_package(Threads REQUIRED)

add_executable(metavision_file_to_video metavision_file_to_video.cpp)
target_link_libraries(metavision_file_to_video
    PRIVATE
        MetavisionSDK::driver
        MetavisionSDK::core
        Boost::program_options Boost::filesystem Threads::Threads
        opencv_core opencv_highgui opencv_videoio
)