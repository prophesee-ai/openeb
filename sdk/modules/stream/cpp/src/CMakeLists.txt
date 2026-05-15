# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_sdk_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_generation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_live.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_offline_raw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_offline_generic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_stream_slicer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dat_event_file_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/erc_counter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_file_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_file_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ext_trigger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_diff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_histo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_event_file_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_event_file_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monitoring.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/offline_streaming_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_event_file_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_event_file_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_evt2_event_file_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_streams_slicer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_system_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synced_camera_system_factory.cpp
)


add_subdirectory(protobuf)

target_sources(metavision_device_serialization_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_serialization.cpp
)
target_include_directories(metavision_device_serialization_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
