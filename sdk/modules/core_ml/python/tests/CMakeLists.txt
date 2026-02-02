# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

lfs_download("datasets/openeb/core_ml" VALIDATION)
lfs_download("sdk/modules/core_ml/models/corner_detection_10_heatmaps.ckpt" VALIDATION)

get_filename_component(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core/python/pypkg" ABSOLUTE)
get_filename_component(CORE_ML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pypkg" ABSOLUTE)

set(PYTHON_MODULES_PATHS ${CORE_DIR} ${CORE_ML_DIR})

add_sdk_python_module_test(core_ml ${PYTHON_MODULES_PATHS})

set_property(TEST pytests_core_ml APPEND PROPERTY ENVIRONMENT "SAMPLES=${CMAKE_CURRENT_SOURCE_DIR}/../samples/")
set_property(TEST pytests_core_ml APPEND PROPERTY ENVIRONMENT "MODELS=${CMAKE_CURRENT_SOURCE_DIR}/../models/")
