# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Include file defining function to easily add modules to Metavision SDK
include(cmake/MetavisionSDKModulesHelper.cmake)

# Include file defining function to easily add test to python packages
include(cmake/MetavisionSDKPythonModuleTest.cmake)

if (ANDROID)
    # Copy Android env.cmake to be available in the build tree
    file(MAKE_DIRECTORY ${MetavisionSDK_CMAKE_OUTPUT_DIR}/android)
    file(COPY ${PROJECT_SOURCE_DIR}/cmake/android/env.cmake
         DESTINATION ${MetavisionSDK_CMAKE_OUTPUT_DIR}/android
    )

    # Install Android env.cmake to be available in the install tree
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/android
            DESTINATION ${MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE}
    )
endif (ANDROID)

# Add the different modules of MetavisionSDK package :
add_subdirectory(modules)
