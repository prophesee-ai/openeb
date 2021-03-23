# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

###########################
#     metavision-sdk-ui   #
###########################

# Runtime (library)
set(CPACK_COMPONENT_METAVISION-SDK-UI_DESCRIPTION "Metavision SDK UI library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-UI_DEPENDS metavision-sdk-base)

# Development package
set(CPACK_COMPONENT_METAVISION-SDK-UI-DEV_DESCRIPTION "Development (C++) files for Metavision SDK UI library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-UI-DEV_DEPENDS metavision-sdk-ui metavision-sdk-base-dev metavision-sdk-core-dev)
list(APPEND CPACK_DEBIAN_METAVISION-SDK-UI-DEV_PACKAGE_DEPENDS "libopencv-dev" "libglfw3-dev" "libboost-dev" "libglew-dev")
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-SDK-UI-DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-SDK-UI-DEV_PACKAGE_DEPENDS}")

# Samples of metavision-sdk-ui
set(CPACK_COMPONENT_METAVISION-SDK-UI-SAMPLES_DESCRIPTION "Samples for Metavision SDK UI library.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-UI-SAMPLES_DEPENDS metavision-sdk-ui-dev metavision-sdk-base-dev)

# Python bindings
set(CPACK_COMPONENT_METAVISION-SDK-UI-PYTHON_DESCRIPTION "Metavision SDK UI Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-UI-PYTHON_DEPENDS metavision-sdk-base metavision-sdk-ui metavision-sdk-base-python metavision-sdk-core-python)