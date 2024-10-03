# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/internal")
    set(BUILD_INTERNAL_PLUGINS TRUE)
    set(metavision_hal_psee_internal_plugins_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/internal/include")
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/internal")

find_file(v4l2_headers videodev2.h PATH_SUFFIXES linux)
message(STATUS "Found V4l2 header : ${v4l2_headers}")

find_file(dma_buf_header dma-buf.h PATH_SUFFIXES linux)
message(STATUS "Found dma-buf header : ${dma_buf_header}")

find_file(dma_heap_header dma-heap.h PATH_SUFFIXES linux)
message(STATUS "Found dma-heap header : ${dma_heap_header}")

set(HAS_V4L2_SUPPORT FALSE)
if(v4l2_headers AND dma_buf_header AND dma_heap_header)
    set(HAS_V4L2_SUPPORT TRUE)
endif()
set(HAS_V4L2 ${HAS_V4L2_SUPPORT} CACHE BOOL "Has V4l2 support with dma")

# Add libraries (plugins)
set(biasgen_calib_files_dir "${GENERATE_FILES_DIRECTORY}/biasgen_calib")
set(metavision_hal_psee_plugins_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(metavision_psee_hw_layer_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/psee_hw_layer_headers/include")
add_subdirectory(lib)

# Add sources plugins
add_subdirectory(src)

# Add internal sources plugins
if(BUILD_INTERNAL_PLUGINS)
    add_subdirectory(internal)
endif(BUILD_INTERNAL_PLUGINS)

# Resources
add_subdirectory(resources)

# Apps
add_subdirectory(apps)

# Samples
add_subdirectory(samples)

# Python
if(COMPILE_PYTHON3_BINDINGS)
    add_subdirectory(python)
endif(COMPILE_PYTHON3_BINDINGS)

# Tests
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)

# Project Raven
add_subdirectory_if_exists(raven)

add_cpack_component(PUBLIC metavision-hal-prophesee-hw-layer-lib metavision-hal-prophesee-hw-layer-dev)
add_cpack_component(PUBLIC metavision-hal-prophesee-plugins)
