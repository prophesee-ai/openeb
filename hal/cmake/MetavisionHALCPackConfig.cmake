# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

###########################
#       metavision-hal    #
###########################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-HAL_DESCRIPTION "Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL_DEPENDS metavision-sdk-base)

############################
#    metavision-hal-dev    #
############################
set(CPACK_COMPONENT_METAVISION-HAL-DEV_DESCRIPTION "Development (C++) files for Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-DEV_DEPENDS metavision-hal metavision-sdk-base-dev)

############################
#    metavision-hal-bin    #
############################
set(CPACK_COMPONENT_METAVISION-HAL-BIN_DESCRIPTION "Metavision HAL applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-BIN_DEPENDS metavision-hal)

############################
#  metavision-hal-samples  #
############################
set(CPACK_COMPONENT_METAVISION-HAL-SAMPLES_DESCRIPTION "Samples for Metavision HAL libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-SAMPLES_DEPENDS metavision-hal-dev)

############################
#   metavision-hal-python  #
############################
set(CPACK_COMPONENT_METAVISION-HAL-PYTHON_DESCRIPTION "Metavision HAL Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-PYTHON_DEPENDS metavision-hal metavision-sdk-base-python)

############################
#    metavision-hal-doc    #
############################
set(CPACK_COMPONENT_METAVISION-HAL-DOC_DESCRIPTION "Documentation of Metavision HAL API.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-DOC_DEPENDS metavision-sdk-base-doc)