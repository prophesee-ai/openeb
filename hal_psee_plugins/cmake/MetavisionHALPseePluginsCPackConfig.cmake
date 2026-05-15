# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

#########################################
# metavision-hal-prophesee-hw-layer-lib #
#########################################

set(CPACK_COMPONENT_METAVISION-HAL-PROPHESEE-HW-LAYER-LIB_DESCRIPTION "Prophesee HW Layer library for Metavision HAL Plugins.\n${PACKAGE_LICENSE}")

#########################################
# metavision-hal-prophesee-hw-layer-dev #
#########################################

set(CPACK_COMPONENT_METAVISION-HAL-PROPHESEE-HW-LAYER-DEV_DESCRIPTION "Prophesee HW Layer headers for Metavision HAL Plugins.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-PROPHESEE-HW-LAYER-DEV_DEPENDS metavision-hal-prophesee-hw-layer-lib)

####################################
# metavision-hal-prophesee-plugins #
####################################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-HAL-PROPHESEE-PLUGINS_DESCRIPTION "Prophesee Plugins for Metavision HAL.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-PROPHESEE-PLUGINS_DEPENDS metavision-hal-prophesee-hw-layer-lib metavision-hal-prophesee-hw-layer-dev)
