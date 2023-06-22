# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(treuzell)
add_subdirectory(gen31)
add_subdirectory(gen41)
add_subdirectory(imx636)
add_subdirectory(imx646)

if(NOT ANDROID)
    add_subdirectory(genx320)
endif(NOT ANDROID)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/saphir")
    add_subdirectory(saphir)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/saphir")

add_subdirectory(psee-video)
add_subdirectory(others)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/sensorlib")
    add_subdirectory(sensorlib)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/sensorlib")
