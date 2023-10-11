# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

include(get_prepended_env_paths)
# we either run this test with plugin from the build tree, or once the plugin has been deployed via DEB packages, in which case, h5py will not 
# find the package because it's not looking in the same default plugin folder (when installed via pip), so we need to specify it in the HDF5_PLUGIN_PATH env. var
get_prepended_env_paths(HDF5_PLUGIN_PATH hdf5_plugin_path "${HDF5_ECF_PLUGIN_BUILD_PATH}" "${HDF5_ECF_PLUGIN_DEB_INSTALL_PATH}")
add_test_app(metavision_file_to_hdf5
    ENV
        "HAS_HDF5=${HDF5_FOUND}"
        "HDF5_PLUGIN_PATH=${hdf5_plugin_path}"
)