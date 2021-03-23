# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Definition of pytest hooks

import pytest
import os

# --- PYTEST HOOKS ---


def pytest_addoption(parser):
    # Define the supported command line arguments
    parser.addoption(
        "--dataset-dir",
        action="store",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets"),
        help="Path to the directory containing the dataset for the tests (stored with git lfs)")
    parser.addoption("--modules", action="store", default="", metavar="COMMA-SEPARATED-LIST",
                     help="only run tests marked as part of the specified modules")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line("markers", "module(name): mark test as part of named module")


def select_listed_modules(items, config):
    modules_list_str = config.getoption("--modules")
    if modules_list_str == "":
        return
    modules_list = modules_list_str.split(",")

    remaining = []
    deselected = []
    for colitem in items:
        if hasattr(colitem, "get_marker"):
            module_marker = colitem.get_marker("module")
        else:
            module_marker = colitem.get_closest_marker("module")
        if module_marker is None or module_marker.args[0] in modules_list:
            remaining.append(colitem)
        else:
            deselected.append(colitem)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def pytest_collection_modifyitems(items, config):
    select_listed_modules(items, config)


def pytest_runtest_setup(item):
    # skip tests whose modules were not listed on the command line
    if hasattr(item, "get_marker"):
        module_marker = item.get_marker("module")
    else:
        module_marker = item.get_closest_marker("module")
    modules_list = item.config.getoption("--modules")
    if modules_list != "" and module_marker is not None:
        module_name = module_marker.args[0]
        if module_name not in modules_list:
            pytest.skip("test is part of skipped module %r" % module_name)


# --- CUSTOM FIXTURES ---

# Retrieves the command line argument 'dataset-dir'
@pytest.fixture
def dataset_dir(request):
    return request.config.getoption("--dataset-dir")
