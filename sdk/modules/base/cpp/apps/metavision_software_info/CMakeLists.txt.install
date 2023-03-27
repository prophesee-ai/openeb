# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_software_info)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(metavision_software_info metavision_software_info.cpp)
target_link_libraries(metavision_software_info PRIVATE MetavisionSDK::base Boost::program_options)