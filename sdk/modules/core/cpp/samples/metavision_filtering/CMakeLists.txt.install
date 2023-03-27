# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_filtering)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

set (sample metavision_filtering)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample} MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui Boost::program_options)
