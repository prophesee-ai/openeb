# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_software_info metavision_software_info.cpp)
target_link_libraries(metavision_software_info PRIVATE MetavisionSDK::base Boost::program_options)

install(TARGETS metavision_software_info
        RUNTIME DESTINATION bin
        COMPONENT metavision-sdk-base-bin
)

install(FILES metavision_software_info.cpp
        DESTINATION share/metavision/sdk/base/cpp_samples/metavision_software_info
        COMPONENT metavision-sdk-base-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/base/cpp_samples/metavision_software_info
        COMPONENT metavision-sdk-base-samples
)

# Test application
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)