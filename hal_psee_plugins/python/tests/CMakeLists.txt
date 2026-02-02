# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

get_pybind_pythonpath(PYTHON3_PYTHON_PATH)

add_pytest(NAME hal-psee-plugins-python-bindings-test
    PATH ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    HAL_PLUGIN_PATH "${PROJECT_BINARY_DIR}/hal/cpp/test/plugins"
    PYTHONPATH ${PYTHON3_PYTHON_PATH}
)
