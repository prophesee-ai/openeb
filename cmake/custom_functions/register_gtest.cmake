# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

include(CMakeParseArguments)

# Function to register gtest executable to ctest
#
# Parameters
#   TEST                       - Name of the test to add
#   TARGET                     - Target to associate with the test
#   HAL_PLUGIN_PATH (optional) - path of the folder with HAL plugin. If not provided, default value is used
#   DATASET (optional)         - path of the folder with custom dataset (by default: ${PROJECT_SOURCE_DIR}/datasets)
#
# Usage:
#   register_gtest(TEST hal-unit-tests TARGET gtest_metavision_hal)
function(register_gtest)
    cmake_parse_arguments(PARSED_ARGS "" "TEST;TARGET;HAL_PLUGIN_PATH;DATASET" "" ${ARGN})

    # Check validity of input args
    foreach(mandatory_arg TEST TARGET)
        if(NOT PARSED_ARGS_${mandatory_arg})
            message(SEND_ERROR "Error when calling function register_gtest : missing mandatory argument ${mandatory_arg} : ${PARSED_ARGS_${mandatory_arg}}")
            return()
        endif(NOT PARSED_ARGS_${mandatory_arg})
    endforeach(mandatory_arg)

    set(hal_plugin_path "${HAL_BUILD_PLUGIN_PATH}")
    if(PARSED_ARGS_HAL_PLUGIN_PATH)
        set(hal_plugin_path "${PARSED_ARGS_HAL_PLUGIN_PATH}")
	endif(PARSED_ARGS_HAL_PLUGIN_PATH)

    if(NOT PARSED_ARGS_DATASET)
        set(DATASET_DIR "${PROJECT_SOURCE_DIR}/datasets")
    else(NOT PARSED_ARGS_DATASET)
        set(DATASET_DIR ${PARSED_ARGS_DATASET})
    endif(NOT PARSED_ARGS_DATASET)

    add_test(
        NAME ${PARSED_ARGS_TEST}
        COMMAND $<TARGET_FILE:${PARSED_ARGS_TARGET}> --dataset-dir "${DATASET_DIR}"  --gtest_color=yes --gtest_output=xml:${GTEST_XML_OUTPUT_DIRECTORY}/${PARSED_ARGS_TEST}.xml
    )
    set_property(TEST ${PARSED_ARGS_TEST} PROPERTY ENVIRONMENT MV_HAL_PLUGIN_PATH=${hal_plugin_path})
endfunction(register_gtest)
