# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

find_package(Threads REQUIRED)

set (sample metavision_simple_window)
set (common_libraries
    MetavisionSDK::ui
    MetavisionSDK::base
    opencv_imgcodecs
    Threads::Threads)

set(FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(constants.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp @ONLY)

add_executable(${sample} ${sample}.cpp constants.h ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp)
target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${sample} PRIVATE ${common_libraries} opencv_highgui)

install(FILES ${sample}.cpp constants.cpp.in constants.h prophesee_logo.jpg
        DESTINATION share/metavision/sdk/ui/cpp_samples/${sample}
        COMPONENT metavision-sdk-ui-samples)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/ui/cpp_samples/${sample}
        COMPONENT metavision-sdk-ui-samples)
