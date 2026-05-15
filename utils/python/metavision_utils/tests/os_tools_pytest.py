
# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from unittest.mock import patch
from os.path import normcase
from metavision_utils import os_tools


@patch('os.getcwd')
def pytestcase_temporary_directory_handler_default_root(mock_getcwd):
    mock_getcwd.return_value = "/a/path/to/cwd/"
    tmp = os_tools.TemporaryDirectoryHandler(create_dir_right_away=False)
    assert normcase(tmp.temporary_directory_root()) == normcase(
        "/a/path/to/cwd/tmp"), "Temporary directory are based on current working dir by default"
