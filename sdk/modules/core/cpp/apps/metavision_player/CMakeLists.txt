# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_player
    src/main.cpp
    src/viewer.cpp
    src/view.cpp
    src/analysis_view.cpp
    src/camera_view.cpp
)
target_link_libraries(metavision_player
    PRIVATE 
        Boost::filesystem
        Boost::program_options
        ${OpenCV_LIBS}
        MetavisionSDK::core
        MetavisionSDK::driver
)
target_include_directories(metavision_player PRIVATE inc)

install(
    TARGETS metavision_player
    RUNTIME DESTINATION bin
    COMPONENT metavision-sdk-core-bin
)

# Install source code
install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/core/apps/metavision_player
        COMPONENT metavision-sdk-core-samples
)
install(DIRECTORY inc src
        DESTINATION share/metavision/sdk/core/apps/metavision_player
        COMPONENT metavision-sdk-core-samples
)

if (BUILD_TESTING)
    add_subdirectory(test)
endif ()