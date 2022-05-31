#!/usr/bin/env python

# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Module with utility functions used in the pytests
"""

import os
import inspect
from metavision_utils.shell_tools import execute_cmd
from metavision_utils.os_tools import TemporaryDirectoryHandler


def run_cmd_setting_mv_log_file(cmd, **kwargs):
    '''
    Runs command given in input and returns error code and output, where the output is the content of
    the MV_LOG_FILE file
    '''

    # Create temporary directory for output log file
    tmp_dir = TemporaryDirectoryHandler()
    output_log_file = os.path.join(tmp_dir.temporary_directory(), "log_output.txt")

    my_env = os.environ.copy()
    my_env["MV_LOG_FILE"] = output_log_file
    _, _, error_code = execute_cmd(cmd, env=my_env, **kwargs)

    # Get output from file
    assert os.path.exists(output_log_file)
    with open(output_log_file, 'r') as file:
        output = file.read()
    return output, error_code


def get_mv_info_stripped_output(res):
    lines = res.splitlines()
    for i in range(len(lines)):
        if lines[i].startswith("="):
            return "\n".join([line.strip() for line in lines[i:]])


def compare_lists(list_1, list_2):
    """"Compares two lists

    Args:
        list_1, list_2 : lists to compare

    Returns:
        two lists : the first containing the elements present in list_1 but not in list_2, the second
                    containing the elements present in list_2 but not list_1

    """
    extra_elements_in_list_1 = []
    for element in list_1:
        if element not in list_2:
            extra_elements_in_list_1.append(element)

    extra_elements_in_list_2 = []
    for element in list_2:
        if element not in list_1:
            extra_elements_in_list_2.append(element)

    return extra_elements_in_list_1, extra_elements_in_list_2


def get_instance_methods(obj):
    """Returns public methods of a object  class"""

    public_methods = []
    for name, value in inspect.getmembers(obj):
        if type(value).__name__ == "instancemethod":
            public_methods.append(name)
    type(value).__name__ == "instancemethod"
    return public_methods


def get_public_classes(obj):
    """Returns public inner classes of a object class"""
    classes = inspect.getmembers(obj, predicate=inspect.isclass)
    public_classes = []
    for name, _ in classes:
        if not name.startswith('_'):
            public_classes.append(name)
    return public_classes
