# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

file(
    MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app/assets
)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${PROJECT_SOURCE_DIR}/datasets/openeb/gen4_evt3_hand.raw
        ${CMAKE_CURRENT_BINARY_DIR}/app/assets/gen4_evt3_hand.raw
)
add_android_app(metavision_viewer_android
    DEPENDS MetavisionSDK::stream hal_plugins
)