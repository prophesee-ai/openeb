# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(HAL_INSTALL_SUPPORT_RELATIVE_PATH "share/metavision/hal")

# Configure file metavision_hal_install_path.h.in. This is done in 2 steps :
#
# 1) at cmake configure time, with configure_file() : this expands the cmake variables inside @...@
#
# 2) at cmake generate time, with file(GENERATE ...) : to set the cmake generator expressions $<...>
#
# The second step is needed to get the output name of target metavision_hal. We need this because when trying to
# automatically determine the installation path under linux we'll look for this file, that we know is always
# installed with package propheee-hal. We do like this instead of simply go look for a directory (like for example
# share/metavision/hal) because if you uninstall from build tree, the directory will remain (although empty), and
# we would detect /usr/local instead of /usr as the installation path
configure_file(metavision_hal_install_path.h.in ${GENERATE_FILES_DIRECTORY}/metavision_hal_install_path.h.in @ONLY)

set(install_paths_infos_dir ${GENERATE_FILES_DIRECTORY}/metavision_hal_install_paths/$<CONFIG>)
file(GENERATE
     OUTPUT ${install_paths_infos_dir}/metavision_hal_install_path.h
     INPUT ${GENERATE_FILES_DIRECTORY}/metavision_hal_install_path.h.in
)

# Generate a header with version number and VCS information
# This header is used by file hal/src/utils/software_info.cpp.
set(VERSION_FILE_RELATIVE_PATH include/metavision/hal/version.h)
set(VERSION_FILE_OUTPUT_PATH ${GENERATE_FILES_DIRECTORY}/${VERSION_FILE_RELATIVE_PATH})

# Add libraries (hal and hal_discovery)
set(metavision_hal_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(lib)

# Add sources to the libraries and plugins
add_subdirectory(src)

# Samples
if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif (BUILD_SAMPLES)

# Tests
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)

add_cpack_component(PUBLIC metavision-hal-bin metavision-hal-samples metavision-hal-lib metavision-hal-dev)

# Documentations
if (GENERATE_DOC)
    add_subdirectory(doc)
endif (GENERATE_DOC)
