# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Register a pytest to the project that will be run by `ctest`
#
# Usege : add_pytest(
#           NAME <pytest_name   : name of the pytest
#           PATH <path1>  : path or directory to your pytests
#           WORKING_DIRECTORY <wd> : working dir in which to execute the tests
#         )
function(add_pytest)
    set(oneValueArgs NAME PATH WORKING_DIRECTORY)
    cmake_parse_arguments(PYTEST_ARGS "" "${oneValueArgs}" "" ${ARGN})
    foreach(arg IN LISTS oneValueArgs)
        if(NOT PYTEST_ARGS_${arg})
            message(FATAL_ERROR "Missing argument '${arg}' in function 'add_pytest()'")
        endif()
    endforeach(arg)
    add_test(
        NAME ${PYTEST_ARGS_NAME}
        COMMAND ${PYTEST_CMD} ${PYTEST_ARGS_PATH} -vv --capture=no --color=yes --junitxml=${JUNIT_XML_OUTPUT_DIRECTORY}/${PYTEST_ARGS_NAME}.xml
        WORKING_DIRECTORY ${PYTEST_ARGS_WORKING_DIRECTORY}
    )
    set_property(TEST ${PYTEST_ARGS_NAME} PROPERTY ENVIRONMENT
        PYTHONPATH=${PROJECT_SOURCE_DIR}/utils/python # to be able to use prophesee_utils python module
        MV_HAL_PLUGIN_PATH=${HAL_BUILD_PLUGIN_PATH}
    )
endfunction(add_pytest)
