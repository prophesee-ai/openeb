# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/metavision_studio/CMakeLists.txt")
    option(COMPILE_METAVISION_STUDIO "Compile Metavision Studio" ON)
    if (COMPILE_METAVISION_STUDIO)
        add_subdirectory(metavision_studio)
    endif ()
endif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/metavision_studio/CMakeLists.txt")

option(COMPILE_PLAYER "Compile Metavision SDK player" ON)
if (COMPILE_PLAYER)
    add_subdirectory(metavision_player)
endif ()
add_subdirectory(metavision_raw_to_video)