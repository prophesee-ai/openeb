# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

target_sources(metavision_hal_psee_plugin_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gen3/gen3_pattern_checkers.cpp
)

# Calib
set(list_calib
    vdac18_8m1.calib
    gen3_idac.calib
    gen3_idac_n_thick.calib
    gen3_idac_n_thin.calib
    gen3_idac_p_thick.calib
    gen3_idac_p_thin.calib
    gen3_idac_railp.calib
)

include(bin2h)
message(STATUS "Generating internal bias files.")
foreach(CALIB_NAME ${list_calib})
    message(STATUS " - ${CALIB_NAME}")
    # Bin2h way, generates a hex array of chars (works on all compilers, but slower to generate).
    string(REGEX REPLACE "\\.[^.]*$" "" CALIB_NAME_NO_EXT ${CALIB_NAME})
    bin2h(SOURCE_FILE "${PROJECT_SOURCE_DIR}/hal_psee_plugins/biasgen/${CALIB_NAME}" VARIABLE_NAME "CALIB")
    configure_file(calib.cc.in "${biasgen_calib_files_dir}/${CALIB_NAME}.cc")
endforeach(CALIB_NAME ${list_calib})
