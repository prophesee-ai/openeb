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

set(MV_GET_STARTED_FILES images README.md)
list_transform_prepend (MV_GET_STARTED_FILES metavision-get-started/)

# Add the files and folders needed to compile open :
foreach (file_or_dir ${MV_GET_STARTED_FILES})
    if (EXISTS "${PROJECT_SOURCE_DIR}/${file_or_dir}")
        get_filename_component(dest "${OUTPUT_DIR}/${file_or_dir}" DIRECTORY)
        file(COPY "${PROJECT_SOURCE_DIR}/${file_or_dir}"
            DESTINATION "${dest}"
            PATTERN __pycache__ EXCLUDE
        )
    endif ()
endforeach(file_or_dir)

set(METAVISION_GET_STARTED_FOLDER "${OUTPUT_DIR}/metavision-get-started/")

function(copy_cpp_sample path)
    set(sample_path ${PROJECT_SOURCE_DIR}/${path})
    if (EXISTS ${sample_path})
        get_filename_component(dest_folder_name ${sample_path} NAME)
        set(dest_path "${METAVISION_GET_STARTED_FOLDER}")
        file(COPY "${sample_path}"
             DESTINATION "${dest_path}"
        )
        set(cmakelist_file ${dest_path}${dest_folder_name}/CMakeLists.txt)
        file (REMOVE ${cmakelist_file})
        file (RENAME ${cmakelist_file}.install ${cmakelist_file})
    else()
        message(FATAL_ERROR "The requested sample does not exist: ${sample_path}")
    endif ()
endfunction()

function(copy_python_sample path exclude_files)
    set(sample_path ${PROJECT_SOURCE_DIR}/${path})
    if (EXISTS ${sample_path})
        get_filename_component(dest_folder_name "${sample_path}" NAME)
        set(dest_path "${METAVISION_GET_STARTED_FOLDER}")
        file(COPY "${sample_path}"
            DESTINATION "${dest_path}"
            PATTERN __pycache__ EXCLUDE
        )
        file (REMOVE ${dest_path}${dest_folder_name}/CMakeLists.txt)
        foreach (file ${exclude_files})
            if (EXISTS "${sample_path}/${file}")
                file (REMOVE "${dest_path}${dest_folder_name}/${file}")
            endif ()
        endforeach(file)
    else()
        message(FATAL_ERROR "The requested sample does not exist: ${sample_path}")
    endif ()
endfunction()

# Add some Metavision Python samples to the metavision-get-started folder
copy_python_sample("sdk/modules/core/python/samples/metavision_time_surface" "")
copy_python_sample("sdk/modules/ml/python/samples/flow_inference" "")

# Add some Metavision C++ samples to the metavision-get-started folder
copy_cpp_sample("sdk/modules/core/cpp/samples/metavision_time_surface")
copy_cpp_sample("sdk/modules/core/cpp/samples/metavision_dummy_radar")

# Copy license files
set(LICENSE_FILES "licensing/LICENSE_METAVISION_SDK" "licensing/LICENSE_OPEN" "licensing/OPEN_SOURCE_3RDPARTY_NOTICES")
foreach (file ${LICENSE_FILES})
    file(COPY "${PROJECT_SOURCE_DIR}/${file}"
        DESTINATION "${METAVISION_GET_STARTED_FOLDER}/licensing"
    )
endforeach(file)

