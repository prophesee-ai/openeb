# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(extra_required_packages "")
if (HDF5_FOUND)
    # not strictly necessary but simplify things, if we switch the dependency to HDF5 ECF from private to public
    set(extra_required_packages "hdf5_ecf")
endif (HDF5_FOUND)

MetavisionSDK_add_module(stream
    REQUIRED_METAVISION_SDK_MODULES
        PUBLIC
            base
            core
    EXTRA_REQUIRED_PACKAGE MetavisionHAL ${extra_required_packages}
)

target_link_libraries(metavision_sdk_stream
    PUBLIC
        metavision_hal
        metavision_hal_discovery
    PRIVATE
        metavision_device_serialization_obj
)
if (HDF5_FOUND)
    target_compile_definitions(metavision_sdk_stream
        PRIVATE
            HAS_HDF5
            ${HDF5_DEFINITIONS}
            ${HDF5_CXX_DEFINITIONS}
    )
    target_include_directories(metavision_sdk_stream
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(metavision_sdk_stream
        PRIVATE
            ${HDF5_LIBRARIES} 
            hdf5_ecf_codec
    )
endif ()