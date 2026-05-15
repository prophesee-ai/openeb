# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Create directory where we will add the files needed to compile the open source offer
file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# For some reason, CMAKE_MODULE_PATH passed by create_metavision_open_archive
# has spaces instead of semicolumns
string(REPLACE " " ";" CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")
string(REPLACE " " ";" HAL_OPEN_PLUGIN_DEVICES "${HAL_OPEN_PLUGIN_DEVICES}")

include(overridden_cmake_functions)

set(HAL_OPEN_PLUGIN_FILES apps biasgen cmake CMakeLists.txt lib resources samples python test)
set(HAL_OPEN_PLUGIN_INCLUDES boards utils geometries plugin devices/common devices/utils devices/others)
set(HAL_OPEN_PLUGIN_HW_LAYER_INCLUDES boards facilities utils devices/common devices/utils devices/psee-video)
set(HAL_OPEN_PLUGIN_SOURCES boards CMakeLists.txt facilities plugin utils devices/common devices/utils devices/psee-video devices/others devices/CMakeLists.txt)

foreach (open_device ${HAL_OPEN_PLUGIN_DEVICES})
    list(APPEND HAL_OPEN_PLUGIN_INCLUDES devices/${open_device})
    list(APPEND HAL_OPEN_PLUGIN_SOURCES devices/${open_device})
    list(APPEND HAL_OPEN_PLUGIN_HW_LAYER_INCLUDES devices/${open_device})
endforeach(open_device)
list_transform_prepend (HAL_OPEN_PLUGIN_INCLUDES include/)
list_transform_prepend (HAL_OPEN_PLUGIN_SOURCES src/)
list_transform_prepend (HAL_OPEN_PLUGIN_HW_LAYER_INCLUDES psee_hw_layer_headers/include/metavision/psee_hw_layer/)
list(APPEND HAL_OPEN_PLUGIN_FILES ${HAL_OPEN_PLUGIN_INCLUDES} ${HAL_OPEN_PLUGIN_SOURCES} ${HAL_OPEN_PLUGIN_HW_LAYER_INCLUDES})
list_transform_prepend (HAL_OPEN_PLUGIN_FILES hal_psee_plugins/)

set(DIRECT_COPY_FILES
  CMakeLists.txt
  licensing/LICENSE_OPEN
  licensing/OPEN_SOURCE_3RDPARTY_NOTICES
  .gitignore conftest.py
  pytest.ini
  cmake
  standalone_samples
  hal
  ${HAL_OPEN_PLUGIN_FILES}
  utils/python/metavision_utils
  utils/python/requirements_openeb.txt
  utils/cpp utils/scripts
  utils/windows/resources.rc.in
  utils/windows/vcpkg-openeb.json
  utils/CMakeLists.txt
  sdk/cmake
  sdk/CMakeLists.txt
  sdk/modules/CMakeLists.txt
)

# Add the files and folders needed to compile open :
foreach (file_or_dir ${DIRECT_COPY_FILES})
    if (EXISTS "${PROJECT_SOURCE_DIR}/${file_or_dir}")
        get_filename_component(dest "${OUTPUT_DIR}/${file_or_dir}" DIRECTORY)
        file(COPY "${PROJECT_SOURCE_DIR}/${file_or_dir}"
            DESTINATION "${dest}"
            PATTERN __pycache__ EXCLUDE
        )
    endif ()
endforeach(file_or_dir)
file(MAKE_DIRECTORY "${OUTPUT_DIR}/.github")
file(COPY "${PROJECT_SOURCE_DIR}/utils/github_actions/openeb/"
     DESTINATION "${OUTPUT_DIR}/.github"
     PATTERN "*"
)
file(COPY "${PROJECT_SOURCE_DIR}/utils/github_actions/openeb/mergify.yml"
     DESTINATION "${OUTPUT_DIR}/.github"
)

file(COPY "${PROJECT_SOURCE_DIR}/utils/jenkins/windows/vcpkg.json"
     DESTINATION "${OUTPUT_DIR}/utils/windows"
)

file(COPY "${PROJECT_SOURCE_DIR}/utils/jenkins/containers/Dockerfile.OpenEB"
     DESTINATION "${OUTPUT_DIR}/utils/containers"
)

# Remove SDK targets
file(REMOVE_RECURSE "${OUTPUT_DIR}/cmake/custom_targets_metavision_sdk")
file(REMOVE "${OUTPUT_DIR}/cmake/custom_functions/create_addon_module_archive_folder.cmake")
file(REMOVE "${OUTPUT_DIR}/cmake/custom_functions/documentation.cmake")
# Remove unwanted files
file(REMOVE "${OUTPUT_DIR}/sdk/cmake/MetavisionSDKCPackConfig.cmake")
file(REMOVE "${OUTPUT_DIR}/sdk/cmake/MetavisionStudioCPackConfig.cmake")
file(REMOVE_RECURSE "${OUTPUT_DIR}/hal/cpp/doc")
file(REMOVE_RECURSE "${OUTPUT_DIR}/hal/python/doc")
foreach (mod base core stream ui core_ml)
    file(COPY "${PROJECT_SOURCE_DIR}/sdk/modules/${mod}" DESTINATION "${OUTPUT_DIR}/sdk/modules")
    # Remove the code we don't want from the SDK modules (doc):
    foreach(subdir doc)
        set(subdirpath "${OUTPUT_DIR}/sdk/modules/${mod}/cpp/${subdir}")
        if(EXISTS "${subdirpath}")
            file(REMOVE_RECURSE "${subdirpath}")
        endif(EXISTS "${subdirpath}")
        set(subdirpath "${OUTPUT_DIR}/sdk/modules/${mod}/python/${subdir}")
        if(EXISTS "${subdirpath}")
            file(REMOVE_RECURSE "${subdirpath}")
        endif(EXISTS "${subdirpath}")
    endforeach(subdir)
endforeach(mod)

if (NOT KEEP_GIT_SUBMODULES)
    # Remove hdf5_ecf submodule
    file(REMOVE_RECURSE "${OUTPUT_DIR}/sdk/modules/stream/cpp/3rdparty/hdf5_ecf")
endif ()
# Remove Metavision Studio :
file(REMOVE_RECURSE "${OUTPUT_DIR}/sdk/modules/core/cpp/apps/metavision_studio")

# Remove evk3d viewer as the plugin is closed source
file(REMOVE_RECURSE "${OUTPUT_DIR}/sdk/modules/stream/cpp/samples/metavision_evk3d_viewer")

# Now we need to modify the way to determine the VCS information (folder created is not be a git repo)
string(CONCAT licence_header "# Copyright (c) Prophesee S.A.\n"
"#\n"
"# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"# you may not use this file except in compliance with the License.\n"
"# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0\n"
"# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed\n"
"# on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"# See the License for the specific language governing permissions and limitations under the License.\n")

file(READ "${OUTPUT_DIR}/cmake/custom_functions/add_library_version_header.cmake" contents)
string(REPLACE "${licence_header}" "" contents_without_header "${contents}")
set (new_contents "")
foreach (git_info BRANCH COMMIT_ID COMMIT_DATE)
    string(REPLACE "\\ " ";" CMD "${GIT_COMMAND_GET_${git_info}}")
    execute_process(
        COMMAND ${CMD}
        OUTPUT_VARIABLE GIT_${git_info}
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Verify that the variable is well defined
    if(NOT GIT_${git_info})
        message(FATAL_ERROR "Could not retrieve GIT ${git_info}" )
    endif(NOT GIT_${git_info})

    set (new_contents "${new_contents}\nset(GIT_${git_info} \"${GIT_${git_info}}\")")
endforeach (git_info)
set(new_contents "${licence_header}${new_contents}\n${contents_without_header}")
file(WRITE "${OUTPUT_DIR}/cmake/custom_functions/add_library_version_header.cmake" "${new_contents}")

# Set GIT_LFS_DOWNLOAD_DRY_RUN to True
file(READ "${OUTPUT_DIR}/cmake/custom_functions/lfs_download.cmake" contents)
string(REPLACE "${licence_header}" "" contents_without_header "${contents}")
file(WRITE "${OUTPUT_DIR}/cmake/custom_functions/lfs_download.cmake" "${licence_header}\nset(GIT_LFS_NOT_AVAILABLE True)\n${contents_without_header}")

# Add README file
file(COPY "${PROJECT_SOURCE_DIR}/cmake/custom_targets/README_metavision_open.md" DESTINATION "${OUTPUT_DIR}")
file(RENAME "${OUTPUT_DIR}/README_metavision_open.md" "${OUTPUT_DIR}/README.md")
