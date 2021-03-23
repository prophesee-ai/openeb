# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Get the list of the registered public components
get_property(components_to_install_public GLOBAL PROPERTY list_cpack_public_components)

###########################
#      metavision-open    #
###########################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-OPEN_DESCRIPTION "Metavision Open libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN_DEPENDS metavision-hal)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module} IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN_DEPENDS metavision-sdk-${available_open_module})
    endif()
endforeach(available_open_module)

############################
#   metavision-open-bin    #
############################
set(CPACK_COMPONENT_METAVISION-OPEN-BIN_DESCRIPTION "Metavision Open applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN-BIN_DEPENDS metavision-hal-bin)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-bin IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN-BIN_DEPENDS metavision-sdk-${available_open_module}-bin)
    endif()
endforeach(available_open_module)

############################
#   metavision-open-dev    #
############################
set(CPACK_COMPONENT_METAVISION-OPEN-DEV_DESCRIPTION "Development (C++) files for Metavision Open libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN-DEV_DEPENDS metavision-hal-dev)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-dev IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN-DEV_DEPENDS metavision-sdk-${available_open_module}-dev)
    endif()
endforeach(available_open_module)


############################
# metavision-open-samples  #
############################
set(CPACK_COMPONENT_METAVISION-OPEN-SAMPLES_DESCRIPTION "Samples for Metavision Open libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN-SAMPLES_DEPENDS metavision-decoders-samples metavision-hal-samples)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-samples IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN-SAMPLES_DEPENDS metavision-sdk-${available_open_module}-samples)
    endif()
endforeach(available_open_module)

############################
#  metavision-open-python  #
############################
set(CPACK_COMPONENT_METAVISION-OPEN-PYTHON_DESCRIPTION "Metavision Open Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN-PYTHON_DEPENDS metavision-hal-python)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-python IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN-PYTHON_DEPENDS metavision-sdk-${available_open_module}-python)
    endif()
endforeach(available_open_module)

####################################
#  metavision-open-python-samples  #
####################################
set(CPACK_COMPONENT_METAVISION-OPEN-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision Open Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPEN-PYTHON-SAMPLES_DEPENDS)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-python-samples IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPEN-PYTHON-SAMPLES_DEPENDS metavision-sdk-${available_open_module}-python-samples)
    endif()
endforeach(available_open_module)


