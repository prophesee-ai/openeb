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
#    Metavision SDK Core - debian packages information
#

# File and package name of the components are automatically set, just need to set the package description
# and potential dependencies

# Runtime (library)
set(CPACK_COMPONENT_METAVISION-SDK-CORE-LIB_DESCRIPTION "Metavision SDK Core library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-LIB_DEPENDS metavision-sdk-base-lib)

# Runtime (apps)
set(CPACK_COMPONENT_METAVISION-SDK-CORE-BIN_DESCRIPTION "Binaries for the Metavision SDK Core applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-BIN_DEPENDS metavision-sdk-core-lib metavision-sdk-stream-lib)

# Development package
set(CPACK_COMPONENT_METAVISION-SDK-CORE-DEV_DESCRIPTION "Development (C++) files for Metavision SDK Core library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-DEV_DEPENDS metavision-sdk-core-lib metavision-sdk-base-dev)
list(APPEND CPACK_DEBIAN_METAVISION-SDK-CORE-DEV_PACKAGE_DEPENDS "libopencv-dev")
list(APPEND CPACK_DEBIAN_METAVISION-SDK-CORE-DEV_PACKAGE_DEPENDS "libboost-dev" "libboost-timer-dev")
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-SDK-CORE-DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-SDK-CORE-DEV_PACKAGE_DEPENDS}")

# Samples of metavision-sdk-core
set(CPACK_COMPONENT_METAVISION-SDK-CORE-SAMPLES_DESCRIPTION "Samples for Metavision SDK Core library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-SAMPLES_DEPENDS metavision-sdk-base-dev metavision-sdk-core-dev metavision-sdk-stream-dev metavision-sdk-ui-dev)

# Pure python library
if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON_DESCRIPTION "Metavision SDK Core Python 3 library.\n${OPEN_PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON_DEPENDS metavision-sdk-core-lib metavision-sdk-base-python${PYTHON3_DEFAULT_VERSION})

    # Python bindings
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
        set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON${py_suffix}_DESCRIPTION "Metavision SDK Core Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
        set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON${py_suffix}_DEPENDS metavision-sdk-core-lib metavision-sdk-base-python${py_suffix})
    endforeach()

    # Python samples of metavision-sdk-core-python
    set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision SDK Core Python 3 library.\n${OPEN_PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-CORE-PYTHON-SAMPLES_DEPENDS metavision-hal-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-core-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-base-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-ui-python${PYTHON3_DEFAULT_VERSION})
endif (COMPILE_PYTHON3_BINDINGS)
