# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if(NOT HAS_V4L2 OR ANDROID)
    return()
endif()
 
target_sources(metavision_psee_hw_layer_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_device_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_ll_biases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_erc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_roi_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_crop.cpp
)
