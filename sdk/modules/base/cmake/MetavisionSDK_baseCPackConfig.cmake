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
#    Metavision SDK Base - debian packages information
#

# File and package name of the components are automatically set, just need to set the package description
# and eventual dependencies

# Runtime (library)
set(CPACK_COMPONENT_METAVISION-SDK-BASE-LIB_DESCRIPTION "Metavision SDK Base library.\n${OPEN_PACKAGE_LICENSE}")

# Runtime (apps)
set(CPACK_COMPONENT_METAVISION-SDK-BASE-BIN_DESCRIPTION "Binaries for the Metavision SDK Base applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-BASE-BIN_DEPENDS metavision-sdk-base-lib)

# Development package
set(CPACK_COMPONENT_METAVISION-SDK-BASE-DEV_DESCRIPTION "Development (C++) files for Metavision SDK Base library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-BASE-DEV_DEPENDS metavision-sdk-base-lib)
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-SDK-BASE-DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-SDK-BASE-DEV_PACKAGE_DEPENDS}")

# Python bindings
if (COMPILE_PYTHON3_BINDINGS)
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
        set(CPACK_COMPONENT_METAVISION-SDK-BASE-PYTHON${py_suffix}_DESCRIPTION "Metavision SDK Base Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
        set(CPACK_COMPONENT_METAVISION-SDK-BASE-PYTHON${py_suffix}_DEPENDS metavision-sdk-base-lib)
      endforeach()
endif (COMPILE_PYTHON3_BINDINGS)

# Samples
set(CPACK_COMPONENT_METAVISION-SDK-BASE-SAMPLES_DESCRIPTION "Samples for Metavision SDK Base library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-BASE-SAMPLES_DEPENDS metavision-sdk-base-dev)