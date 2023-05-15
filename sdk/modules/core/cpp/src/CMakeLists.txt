# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_sdk_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_rate_events_splitter_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cd_frame_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cv_video_recorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_synchronizer_from_triggers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_buffer_reslicer_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fast_math_functions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/misc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/on_demand_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/periodic_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rate_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_event_frame_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_displayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/threaded_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/time_decay_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/video_writer.cpp
)
