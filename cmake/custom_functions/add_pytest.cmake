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
# Usage : add_pytest(
#           NAME <pytest_name   : name of the pytest
#           PATH <path1>  : path or directory to your pytests
#           WORKING_DIRECTORY <wd> : working dir in which to execute the tests
#           HAL_PLUGIN_PATH <plugin_path> : Optional - override path to HAL plugins libraries
#           PYTHONPATH <path1>[;<path2>]... : Optional - directories to be added to the python path
#           ENV <var1=val1;var2=val2;...;varN=valN> : Optional - environment variables defined for the tests
#         )

include(get_prepended_env_paths)

function(add_pytest)
    set(oneValueMandatoryArgs NAME PATH WORKING_DIRECTORY)
    set(oneValueOptionalArgs HAL_PLUGIN_PATH)

    set(oneValueArgs ${oneValueMandatoryArgs} ${oneValueOptionalArgs})
    set(multiValueArgs PYTHONPATH ENV )
    cmake_parse_arguments(PYTEST_ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(arg IN LISTS oneValueMandatoryArgs)
        if(NOT PYTEST_ARGS_${arg})
            message(FATAL_ERROR "Missing argument '${arg}' in function 'add_pytest()'")
        endif()
    endforeach(arg)
    add_test(
        NAME ${PYTEST_ARGS_NAME}
        COMMAND ${PYTEST_CMD} ${PYTEST_ARGS_PATH} -vv --capture=no --color=yes --junitxml=${JUNIT_XML_OUTPUT_DIRECTORY}/${PYTEST_ARGS_NAME}.xml
        WORKING_DIRECTORY ${PYTEST_ARGS_WORKING_DIRECTORY}
    )

    get_prepended_env_paths(PYTHONPATH python_path_val 
        "${PROJECT_SOURCE_DIR}/utils/python" 
        "${PYTEST_ARGS_PYTHONPATH}")

    set(MV_HAL_PLUGIN_PATH ${HAL_BUILD_PLUGIN_PATH})
    if(PYTEST_ARGS_HAL_PLUGIN_PATH)
        set(MV_HAL_PLUGIN_PATH ${PYTEST_ARGS_HAL_PLUGIN_PATH})
    endif()
    
    set_property(TEST ${PYTEST_ARGS_NAME} PROPERTY ENVIRONMENT
        "PYTHONPATH=${python_path_val}"
        "MV_HAL_PLUGIN_PATH=${MV_HAL_PLUGIN_PATH}"
    )

    if(WIN32)
        # add DLLs & executables to windows path
        get_prepended_env_paths(PATH path_value "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
        set_property(TEST ${PYTEST_ARGS_NAME} APPEND PROPERTY ENVIRONMENT "PATH=${path_value}")
    endif(WIN32)

    if (PYTEST_ARGS_ENV)
        set_property(TEST ${PYTEST_ARGS_NAME} APPEND PROPERTY ENVIRONMENT ${PYTEST_ARGS_ENV} APPEND)
    endif (PYTEST_ARGS_ENV)
endfunction(add_pytest)
