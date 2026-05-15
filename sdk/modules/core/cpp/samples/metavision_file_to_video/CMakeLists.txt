# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_file_to_video metavision_file_to_video.cpp)
target_link_libraries(metavision_file_to_video
    PRIVATE
        MetavisionSDK::stream
        MetavisionSDK::core
        Boost::program_options Threads::Threads
        opencv_core opencv_highgui opencv_videoio
)

install(TARGETS metavision_file_to_video
        RUNTIME DESTINATION bin
        COMPONENT metavision-sdk-core-bin
)

install(FILES metavision_file_to_video.cpp
        DESTINATION share/metavision/sdk/core/cpp_samples/metavision_file_to_video
        COMPONENT metavision-sdk-core-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/core/cpp_samples/metavision_file_to_video
        COMPONENT metavision-sdk-core-samples
)

# Test application
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)
