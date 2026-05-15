# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(HAL_PSEE_OPEN_PLUGIN_DEVICES gen31 gen41 imx636 imx646 imx8 genx320 treuzell v4l2
    CACHE INTERNAL "Hal plugin open sourced devices")

set(HAL_PSEE_CLOSED_PLUGIN_DEVICES saphir sensorlib evk3d geny esp_proto
    CACHE INTERNAL "Hal plugin close sourced devices")

set(HAL_PSEE_PLUGIN_DEVICES_COMMON utils common psee-video others)
foreach (hal_device IN LISTS HAL_PSEE_PLUGIN_DEVICES_COMMON HAL_PSEE_OPEN_PLUGIN_DEVICES)
    add_subdirectory(${hal_device})
endforeach()

foreach (hal_closed_device ${HAL_PSEE_CLOSED_PLUGIN_DEVICES})
    add_subdirectory_if_exists(${hal_closed_device})
endforeach()
