# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import os
import platform
import pytest
from metavision_utils import shell_tools

@pytest.mark.skip(reason="Temporarily disabled for v5.1.1")
def pytestcase_deprecated_function_warning_at_compilation():
    cmake_target = "deprecation_warning_sample"
    cmake_build_path = os.getenv("CMAKE_BINARY_DIR")
    cmake_build_type = os.getenv("CMAKE_BUILD_TYPE")
    cmd = "cmake --build {} --target {} --config {}".format(cmake_build_path, cmake_target, cmake_build_type)

    output, error, error_code = shell_tools.execute_cmd(cmd, verbose=True)
    if platform.system() == "Windows":
        assert "warning C4996" in output
    else:
        assert "-Wdeprecated-declarations" or "-Werror=deprecated-declarations" in output
    assert error_code != 0
