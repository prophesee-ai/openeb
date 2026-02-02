# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_raw_evt_encoder metavision_raw_evt_encoder.cpp)
target_link_libraries(metavision_raw_evt_encoder PRIVATE MetavisionSDK::core MetavisionSDK::stream Boost::program_options)

install(FILES metavision_raw_evt_encoder.cpp
        DESTINATION share/metavision/sdk/stream/cpp_samples/metavision_raw_evt_encoder
        COMPONENT metavision-sdk-stream-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/sdk/stream/cpp_samples/metavision_raw_evt_encoder
        COMPONENT metavision-sdk-stream-samples
)
