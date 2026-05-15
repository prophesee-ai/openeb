# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

MetavisionSDK_add_module(base)

target_include_directories(metavision_sdk_base
    PUBLIC
        $<BUILD_INTERFACE:${GENERATE_FILES_DIRECTORY}/include>
        $<INSTALL_INTERFACE:include>
)

if(ANDROID)
    target_link_libraries(metavision_sdk_base PUBLIC log)
endif(ANDROID)

add_dependencies(metavision_sdk_base generate_metavision_sdk_version_header)