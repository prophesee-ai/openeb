# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set (sample metavision_dummy_radar)
set (common_libraries
        MetavisionSDK::core
        MetavisionSDK::stream
        MetavisionSDK::ui
        Boost::program_options
        opencv_core
        opencv_imgproc
        opencv_imgcodecs)

add_executable(${sample} ${sample}.cpp activity_monitoring.cpp radar_viewer.cpp)
target_link_libraries(${sample} PRIVATE ${common_libraries})

install(FILES ${sample}.cpp activity_monitoring.cpp radar_viewer.cpp activity_monitoring.h radar_viewer.h
        DESTINATION share/metavision/sdk/core/cpp_samples/${sample}
        COMPONENT metavision-sdk-core-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/core/cpp_samples/${sample}
        COMPONENT metavision-sdk-core-samples
)
