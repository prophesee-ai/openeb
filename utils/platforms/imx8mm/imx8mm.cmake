# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

message(STATUS "Building for platforms imx8")

# options specific to the platform

## Remove all advanced plugin from the build.
# We need Eigen3 lib compiled as part of the SDK sysroot to enable them
set(METAVISION_SDK_MODULES_ADVANCED CACHE STRING "SDK Advanced modules")

set(HDF5_DISABLED ON CACHE BOOL "disable HDF5")

set(COMPILE_PYTHON3_BINDINGS OFF CACHE BOOL "disable python binding while waiting for pybind 2.6")
option(COMPILE_METAVISION_STUDIO "Compile Metavision Studio" OFF)


execute_process(
    # List all python3 versions available on the targeted platform
    COMMAND sh -c "ls -d1 $ENV{SDKTARGETSYSROOT}/usr/lib/python3*" # SDKTARGETSYSROOT is defined by Yocto env setup script
    OUTPUT_VARIABLE PYTHON_TARGET_LIB_VERSIONS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

foreach(python_target_lib ${PYTHON_TARGET_LIB_VERSIONS})
    # Retrieve the SOABI mane for each available python3 version.
    # Use the SOABI to build up Python bindings shared object suffix.
    string(REGEX MATCH "python3\.[0-9]+" PYTHON_TARGET_VERSION "${python_target_lib}")
    string(REPLACE "python" "" PYTHON_TARGET_SHORT_VERSION ${PYTHON_TARGET_VERSION})
    execute_process(
        COMMAND sh -c "find ${python_target_lib} -name Makefile | xargs egrep '^SOABI' | cut -f3"
        OUTPUT_VARIABLE PYTHON_TARGET_SOABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(PYTHON_TARGET_EXTENSION_SUFFIX ".${PYTHON_TARGET_SOABI}.so")
    set(PYTHON_${PYTHON_TARGET_SHORT_VERSION}_MODULE_EXTENSION "${PYTHON_TARGET_EXTENSION_SUFFIX}" 
        CACHE STRING "Python library suffix for ${PYTHON_TARGET_VERSION}")
endforeach()

