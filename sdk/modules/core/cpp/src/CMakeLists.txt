# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_sdk_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/adaptive_rate_events_splitter_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/base_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/contrast_map_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/event_buffer_reslicer_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/events_integration_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/on_demand_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/periodic_frame_generation_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/time_decay_frame_generation_algorithm.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessors/json_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessors/event_preprocessor_type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessors/tensor.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cd_frame_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cv_video_recorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/data_synchronizer_from_triggers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fast_math_functions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/misc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/rate_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/raw_event_frame_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/threaded_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/video_writer.cpp
)
