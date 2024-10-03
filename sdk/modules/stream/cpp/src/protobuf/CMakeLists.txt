# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_library(metavision_device_serialization_obj OBJECT)
# The following line is needed because when linking a shared library to an object one,
# the object library needs to be compiled with -fPIC
# cf https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
set_target_properties(metavision_device_serialization_obj
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(metavision_device_serialization_obj
    PRIVATE
        metavision_hal
)

if(USE_PROTOBUF)
    find_package(Protobuf CONFIG QUIET)
    if (NOT Protobuf_FOUND)
        find_package(Protobuf REQUIRED)
    endif(NOT Protobuf_FOUND)

    set(PROTO_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/device_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/antiflicker_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_synchronization_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/digital_crop_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/digital_event_mask_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/event_rate_control_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/event_rate_activity_filter_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/event_trail_filter_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/hw_register_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/ll_biases_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/region_of_interest_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/trigger_in_state.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/trigger_out_state.proto
    )

    if (COMMAND protobuf_generate)
      target_sources(metavision_device_serialization_obj PRIVATE ${PROTO_SOURCES})
      protobuf_generate(TARGET metavision_device_serialization_obj
                        LANGUAGE cpp
                        OUT_VAR DEVICE_PROTO_HDRS)
    else()
      protobuf_generate_cpp(DEVICE_PROTO_SRCS DEVICE_PROTO_HDRS ${PROTO_SOURCES})
      target_sources(metavision_device_serialization_obj PRIVATE ${DEVICE_PROTO_SRCS})
    endif()

    foreach(header ${DEVICE_PROTO_HDRS})
        get_filename_component(header_dir ${header} DIRECTORY)
        list(APPEND PROTO_HDRS_DIRS ${header_dir})
    endforeach(header)

    target_compile_definitions(metavision_device_serialization_obj PRIVATE HAS_PROTOBUF)
    target_include_directories(metavision_device_serialization_obj
        PRIVATE
            ${PROTO_HDRS_DIRS}
            ${Protobuf_INCLUDE_DIRS}
    )

    # Doesn't exist for all CMake versions
    if (TARGET protobuf::libprotobuf)
        set(Protobuf_LIBRARIES protobuf::libprotobuf)
    endif ()

    target_link_libraries(metavision_device_serialization_obj
        PRIVATE
            ${Protobuf_LIBRARIES}
    )
endif(USE_PROTOBUF)
