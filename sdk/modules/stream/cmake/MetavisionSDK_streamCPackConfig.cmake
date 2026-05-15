# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

##########################################################
#
#    Metavision SDK Stream - debian packages information
#

# File and package name of the components are automatically set, just need to set the package description
# and potential dependencies

# Runtime (library)
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-LIB_DESCRIPTION "Metavision SDK stream library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-LIB_DEPENDS metavision-hal-lib metavision-sdk-base-lib metavision-sdk-core-lib)
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-LIB_PACKAGE_DEPENDS "libprotobuf-dev")
if (HDF5_FOUND)
    list(APPEND CPACK_DEBIAN_METAVISION-SDK-STREAM-LIB_PACKAGE_DEPENDS "hdf5-ecf-codec-lib")
endif (HDF5_FOUND)
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-SDK-STREAM-LIB_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-SDK-STREAM-LIB_PACKAGE_DEPENDS}")

# Runtime (apps)
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-BIN_DESCRIPTION "Binaries for the Metavision SDK stream applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-BIN_DEPENDS metavision-sdk-stream-lib metavision-sdk-core-lib metavision-hal-prophesee-hw-layer-lib)

# Development package
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-DEV_DESCRIPTION "Development (C++) files for Metavision SDK stream library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-DEV_DEPENDS metavision-sdk-stream-lib metavision-sdk-base-dev metavision-sdk-core-dev metavision-hal-dev)
if (HDF5_FOUND)
    list(APPEND CPACK_DEBIAN_METAVISION-SDK-STREAM-DEV_PACKAGE_DEPENDS "hdf5-ecf-codec-dev")
endif (HDF5_FOUND)
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-SDK-STREAM-DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-SDK-STREAM-DEV_PACKAGE_DEPENDS}")

# Samples
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-SAMPLES_DESCRIPTION "Samples for Metavision SDK stream library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-STREAM-SAMPLES_DEPENDS
    metavision-hal-prophesee-hw-layer-dev
    metavision-sdk-base-dev
    metavision-sdk-core-dev
    metavision-sdk-stream-dev
    metavision-sdk-ui-dev)


if (COMPILE_PYTHON3_BINDINGS)
    # Python bindings
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
        set(CPACK_COMPONENT_METAVISION-SDK-STREAM-PYTHON${py_suffix}_DESCRIPTION "Metavision SDK Core Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
        set(CPACK_COMPONENT_METAVISION-SDK-STREAM-PYTHON${py_suffix}_DEPENDS metavision-sdk-stream-lib metavision-sdk-base-python${py_suffix} metavision-hal-python${py_suffix})
    endforeach()

    # Python samples of metavision-sdk-stream-python
    set(CPACK_COMPONENT_METAVISION-SDK-STREAM-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision SDK Stream Python 3 library.\n${OPEN_PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-STREAM-PYTHON-SAMPLES_DEPENDS metavision-sdk-core-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-base-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-stream-python${PYTHON3_DEFAULT_VERSION})
endif (COMPILE_PYTHON3_BINDINGS)
