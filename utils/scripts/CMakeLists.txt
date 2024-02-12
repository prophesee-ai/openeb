# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Linux environment setup script
set(PYPKG_DIRECTORIES "")
set(_PYPKG_DIRECTORIES)
foreach(module ${METAVISION_SDK_MODULES_AVAILABLE})
    set(module_pypkg "${PROJECT_SOURCE_DIR}/sdk/modules/${module}/python/pypkg")
    if (EXISTS "${module_pypkg}")
        set(PYPKG_DIRECTORIES "${module_pypkg}:${PYPKG_DIRECTORIES}")

        file(TO_NATIVE_PATH "${module_pypkg}" native_module_pypkg)
        set(_PYPKG_DIRECTORIES "${native_module_pypkg};${_PYPKG_DIRECTORIES}")
    endif()
endforeach()

if(UNIX)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/setup_env.sh.in"
        "${GENERATE_FILES_DIRECTORY}/setup_env.sh"
        @ONLY
    )
    file(COPY "${GENERATE_FILES_DIRECTORY}/setup_env.sh"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
endif()

# Windows environment setup script
if(WIN32)
    file(TO_NATIVE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}" _CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}" _CMAKE_SOURCE_DIR)
    file(TO_NATIVE_PATH "${HAL_BUILD_PLUGIN_PATH}" _HAL_BUILD_PLUGIN_PATH)
    file(TO_NATIVE_PATH "${HDF5_ECF_PLUGIN_BUILD_PATH}" _HDF5_ECF_PLUGIN_BUILD_PATH)
    file(TO_NATIVE_PATH "${PYTHON3_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}" _PYTHON3_OUTPUT_DIR)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/setup_env.bat.in"
        "${GENERATE_FILES_DIRECTORY}/setup_env.bat"
        @ONLY
    )
    file(COPY "${GENERATE_FILES_DIRECTORY}/setup_env.bat"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
endif()
