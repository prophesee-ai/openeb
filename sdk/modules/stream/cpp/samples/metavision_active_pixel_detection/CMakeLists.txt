# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set (sample metavision_active_pixel_detection)
set (common_libraries MetavisionSDK::stream MetavisionSDK::core Boost::program_options opencv_highgui)

add_executable(${sample} ${sample}.cpp $<TARGET_OBJECTS:metavision_psee_hw_layer_obj>)
target_include_directories(${sample} PRIVATE metavision_psee_hw_layer_obj)
target_link_libraries(${sample} PRIVATE ${common_libraries} metavision_psee_hw_layer_obj)

install(TARGETS ${sample}
        RUNTIME DESTINATION bin
        COMPONENT metavision-sdk-stream-bin
)

install(FILES ${sample}.cpp
        DESTINATION share/metavision/sdk/stream/cpp_samples/${sample}
        COMPONENT metavision-sdk-stream-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/stream/cpp_samples/${sample}
        COMPONENT metavision-sdk-stream-samples
)