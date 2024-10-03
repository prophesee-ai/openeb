# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

###########################
#    metavision-hal-lib   #
###########################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-HAL-LIB_DESCRIPTION "Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-LIB_DEPENDS metavision-sdk-base-lib)

############################
#    metavision-hal-dev    #
############################
set(CPACK_COMPONENT_METAVISION-HAL-DEV_DESCRIPTION "Development (C++) files for Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-DEV_DEPENDS metavision-hal-lib metavision-sdk-base-dev)

############################
#    metavision-hal-bin    #
############################
set(CPACK_COMPONENT_METAVISION-HAL-BIN_DESCRIPTION "Metavision HAL applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-BIN_DEPENDS metavision-hal-lib)

############################
#  metavision-hal-samples  #
############################
set(CPACK_COMPONENT_METAVISION-HAL-SAMPLES_DESCRIPTION "Samples for Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-SAMPLES_DEPENDS metavision-hal-dev)
set(CPACK_COMPONENT_METAVISION-HAL-SAMPLES_PACKAGE_DEPENDS "libusb-1.0" "libusb-1.0-0-dev")

###################################
#  metavision-hal-python-samples  #
###################################
set(CPACK_COMPONENT_METAVISION-HAL-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision HAL Python libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-PYTHON-SAMPLES_DEPENDS metavision-hal-python${PYTHON3_DEFAULT_VERSION})

############################
# metavision-hal-python3.X #
############################
if (COMPILE_PYTHON3_BINDINGS)
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
      set(CPACK_COMPONENT_METAVISION-HAL-PYTHON${py_suffix}_DESCRIPTION "Metavision HAL Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
      set(CPACK_COMPONENT_METAVISION-HAL-PYTHON${py_suffix}_DEPENDS metavision-hal-lib metavision-sdk-base-python${py_suffix})
    endforeach()
endif (COMPILE_PYTHON3_BINDINGS)
