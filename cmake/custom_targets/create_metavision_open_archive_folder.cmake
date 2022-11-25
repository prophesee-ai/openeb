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
include(overridden_cmake_functions)

set(HAL_OPEN_PLUGIN_FILES apps biasgen cmake CMakeLists.txt lib resources test)
set(HAL_OPEN_INCLUDES boards decoders facilities geometries plugin utils devices/common devices/utils)
set(HAL_OPEN_SOURCES boards CMakeLists.txt facilities plugin utils devices/common devices/utils devices/CMakeLists.txt)
set(HAL_OPEN_DEVICES gen3 gen31 gen41 golden_fallbacks imx636 others treuzell)

foreach (open_device ${HAL_OPEN_DEVICES})
    list(APPEND HAL_OPEN_INCLUDES devices/${open_device})
    list(APPEND HAL_OPEN_SOURCES devices/${open_device})
endforeach(open_device)
list_transform_prepend (HAL_OPEN_INCLUDES include/)
list_transform_prepend (HAL_OPEN_SOURCES src/)
list(APPEND HAL_OPEN_PLUGIN_FILES ${HAL_OPEN_INCLUDES} ${HAL_OPEN_SOURCES})
list_transform_prepend (HAL_OPEN_PLUGIN_FILES hal_psee_plugins/)

# Add the files and folders needed to compile open :
foreach (file_or_dir CMakeLists.txt licensing/LICENSE_OPEN .gitignore conftest.py pytest.ini cmake standalone_samples hal ${HAL_OPEN_PLUGIN_FILES} utils/python/metavision_utils utils/cpp utils/scripts utils/CMakeLists.txt sdk/cmake sdk/CMakeLists.txt sdk/modules/CMakeLists.txt)
    get_filename_component(dest "${OUTPUT_DIR}/${file_or_dir}" DIRECTORY)
    file(COPY "${PROJECT_SOURCE_DIR}/${file_or_dir}"
         DESTINATION "${dest}"
         PATTERN __pycache__ EXCLUDE
    )
endforeach(file_or_dir)
file(MAKE_DIRECTORY "${OUTPUT_DIR}/.github")
file(COPY "${PROJECT_SOURCE_DIR}/utils/github_actions/openeb/"
     DESTINATION "${OUTPUT_DIR}/.github"
     PATTERN "*"
)

# Remove SDK targets
file(REMOVE_RECURSE "${OUTPUT_DIR}/cmake/custom_targets_metavision_sdk")
file(REMOVE "${OUTPUT_DIR}/cmake/custom_functions/create_addon_module_archive_folder.cmake")
file(REMOVE "${OUTPUT_DIR}/cmake/custom_functions/documentation.cmake")
# Remove unwanted plugins
file(REMOVE "${OUTPUT_DIR}/hal_psee_plugins/src/plugin/psee_gen4_evk1.cpp")
file(REMOVE "${OUTPUT_DIR}/hal_psee_plugins/src/plugin/psee_gen4_evk2.cpp")
file(REMOVE_RECURSE "${OUTPUT_DIR}/hal_psee_plugins/src/plugin/raven")
# Remove unwanted files
file(REMOVE "${OUTPUT_DIR}/sdk/cmake/MetavisionSDKCPackConfig.cmake")
file(REMOVE "${OUTPUT_DIR}/sdk/cmake/MetavisionStudioCPackConfig.cmake")
file(REMOVE_RECURSE "${OUTPUT_DIR}/hal/cpp/doc")
file(REMOVE_RECURSE "${OUTPUT_DIR}/hal/python/doc")
foreach (mod base core driver ui core_ml)
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
# Remove Metavision Studio :
file(REMOVE_RECURSE "${OUTPUT_DIR}/sdk/modules/core/cpp/apps/metavision_studio")

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
