# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if (BUILD_TESTING)
    # Utils to write gtests
    add_subdirectory(gtest)
endif (BUILD_TESTING)
if(COMPILE_PYTHON3_BINDINGS)
    # Utils to write python bindings
    add_subdirectory(pybind)
endif(COMPILE_PYTHON3_BINDINGS)

# Utils to instrument and profile code
add_subdirectory(profiling)