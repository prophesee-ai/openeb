# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

lfs_download("datasets/openeb/core/event_io/csv_zip_eth" VALIDATION)
lfs_download("datasets/openeb/blinking_gen4_with_ext_triggers.raw" VALIDATION)
lfs_download("datasets/openeb/diff3d.raw" VALIDATION)
lfs_download("datasets/openeb/histo3d.raw" VALIDATION)
lfs_download("datasets/openeb/histo3d_padding.raw" VALIDATION)
lfs_download("datasets/openeb/events_big_time_gap_repeated_ts.raw" VALIDATION)
lfs_download("datasets/openeb/events_big_time_gap_repeated_ts.dat" VALIDATION)
lfs_download("datasets/openeb/ts_offset_test.hdf5" VALIDATION)

add_sdk_python_module_test(core)

set(HDF5_ECF_PLUGIN_BUILD_PATH "${CMAKE_BINARY_DIR}/lib/hdf5/plugin")
get_prepended_env_paths(HDF5_PLUGIN_PATH hdf5_plugin_path "${HDF5_ECF_PLUGIN_BUILD_PATH}" "${HDF5_ECF_PLUGIN_DEB_INSTALL_PATH}")
set_property(TEST pytests_core APPEND PROPERTY ENVIRONMENT "HDF5_PLUGIN_PATH=${hdf5_plugin_path}")