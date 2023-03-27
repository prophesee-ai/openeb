# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_hal_showcase_sample)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(MetavisionHAL REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV COMPONENTS core highgui REQUIRED)
find_package(Threads REQUIRED)

add_executable(metavision_hal_showcase metavision_hal_showcase.cpp)

target_link_libraries(metavision_hal_showcase PRIVATE Metavision::HAL_discovery Boost::program_options ${OpenCV_LIBS} Threads::Threads)
