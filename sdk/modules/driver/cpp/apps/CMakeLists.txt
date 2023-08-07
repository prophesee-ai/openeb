# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_subdirectory(metavision_file_cutter)
add_subdirectory(metavision_file_info)
add_subdirectory(metavision_file_to_dat)
add_subdirectory(metavision_viewer)
if (HDF5_FOUND)
    add_subdirectory(metavision_file_to_hdf5)
endif ()
if (NOT ANDROID)
    # need support for std::filesystem in NDK
    # AK comment here
    add_subdirectory(metavision_active_pixel_detection)
endif (NOT ANDROID)
