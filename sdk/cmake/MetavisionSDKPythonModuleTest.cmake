# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

include(get_prepended_env_paths)
#####################################################################
#
# Adds a test for a metavision python module
#
#
# usage :
#     add_sdk_python_module_test module_name(<module-name>
#          [[ <additional python dependencies to add to pythonpaths>
#          ]]
#      )
#
#
#
#
# first arg is the name of the module, the function must be called within the test folder of that python module
# assuming the test folder and the module have the same parent folder.
# additionnal arguments are passed to the pythonpath
function(add_sdk_python_module_test module_name)
    get_filename_component(PYTESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

    set(test_name "pytests_${module_name}")
    if(WIN32)
        set(pythonpath_value "${PYTHON3_OUTPUT_DIR}/$<CONFIG>")
    else()
        set(pythonpath_value "${PYTHON3_OUTPUT_DIR}")
    endif()
    get_prepended_env_paths(PYTHONPATH pythonpath_value "${pythonpath_value}" "${PROJECT_SOURCE_DIR}/utils/python")

    add_test(
        NAME ${test_name}
        COMMAND ${PYTEST_CMD} ${PYTESTS_DIR} -vv --capture=no --color=yes --junitxml=${JUNIT_XML_OUTPUT_DIRECTORY}/pytests_${module_name}.xml
        WORKING_DIRECTORY ${PYTESTS_DIR}/..
    )
    # add module dependencies and global utils to the python paths
    set(pypkg_module_dir "${PROJECT_SOURCE_DIR}/sdk/modules/${module_name}/python/pypkg")
    if(EXISTS "${pypkg_module_dir}")
        get_prepended_env_paths(PYTHONPATH pythonpath_value "${pythonpath_value}" "${pypkg_module_dir}")
    endif(EXISTS "${pypkg_module_dir}")

    # put extra arguments in the python paths as well
    if (ARGC GREATER 1)
        set(extra_pythonpath_dependencies ${ARGV})
        list(REMOVE_AT extra_pythonpath_dependencies 0)
        get_prepended_env_paths(PYTHONPATH pythonpath_value "${pythonpath_value}" "${extra_pythonpath_dependencies}")
    endif()
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT
        "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "PYTHONPATH=${pythonpath_value}"
        "MV_HAL_PLUGIN_PATH=${PROJECT_BINARY_DIR}/${HAL_INSTALL_PLUGIN_RELATIVE_PATH}"
    )

    if(WIN32)
        # add DLLs & executables to windows path
        get_prepended_env_paths(PATH path_value "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PATH=${path_value}")
    else(WIN32)
        # add executables to path
        get_prepended_env_paths(PATH path_value "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PATH=${path_value}")
    endif(WIN32)
endfunction(add_sdk_python_module_test)
