# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/devices/gen4")
    set(HAL_GEN4_SUPPORT TRUE)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/devices/gen4")

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/devices/raven")
    set(HAL_RAVEN_SUPPORT TRUE)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/devices/raven")

# Add libraries (plugins)
set(biasgen_calib_files_dir "${GENERATE_FILES_DIRECTORY}/biasgen_calib")
set(metavision_hal_psee_plugins_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(lib)

# Add sources plugins
add_subdirectory(src)

# Resources
add_subdirectory(resources)

# Apps
add_subdirectory(apps)

# Tests
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)

add_cpack_component(PUBLIC metavision-hal-prophesee-plugins)