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

#############################
#    metavision-openeb-lib  #
#############################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-OPENEB-LIB_DESCRIPTION "Metavision OpenEB libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPENEB-LIB_DEPENDS metavision-hal-lib metavision-hal-prophesee-plugins)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-lib IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-LIB_DEPENDS metavision-sdk-${available_open_module}-lib)
    endif()
endforeach(available_open_module)

##############################
#   metavision-openeb-bin    #
##############################
set(CPACK_COMPONENT_METAVISION-OPENEB-BIN_DESCRIPTION "Metavision OpenEB applications.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPENEB-BIN_DEPENDS metavision-openeb-lib metavision-hal-bin)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-bin IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-BIN_DEPENDS metavision-sdk-${available_open_module}-bin)
    endif()
endforeach(available_open_module)

##############################
#   metavision-openeb-dev    #
##############################
set(CPACK_COMPONENT_METAVISION-OPENEB-DEV_DESCRIPTION "Development (C++) files for Metavision OpenEB libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPENEB-DEV_DEPENDS metavision-openeb-lib metavision-hal-dev)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-dev IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-DEV_DEPENDS metavision-sdk-${available_open_module}-dev)
    endif()
endforeach(available_open_module)


##############################
# metavision-openeb-samples  #
##############################
set(CPACK_COMPONENT_METAVISION-OPENEB-SAMPLES_DESCRIPTION "Samples for Metavision OpenEB libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPENEB-SAMPLES_DEPENDS metavision-openeb-dev metavision-hal-samples)
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
    if(metavision-sdk-${available_open_module}-samples IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-SAMPLES_DEPENDS metavision-sdk-${available_open_module}-samples)
    endif()
endforeach(available_open_module)

#################################
#  metavision-openeb-python3.X  #
#################################
if (COMPILE_PYTHON3_BINDINGS)
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
        set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON${py_suffix}_DESCRIPTION "Metavision OpenEB Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
        set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON${py_suffix}_DEPENDS metavision-openeb-lib metavision-hal-python${py_suffix})
        foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
           if(metavision-sdk-${available_open_module}-python${py_suffix} IN_LIST components_to_install_public)
               list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-PYTHON${py_suffix}_DEPENDS metavision-sdk-${available_open_module}-python${py_suffix})
           endif()
        endforeach(available_open_module)
    endforeach()
endif (COMPILE_PYTHON3_BINDINGS)

#############################
#  metavision-openeb-python #
#############################
if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON_DESCRIPTION "Metavision OpenEB Python 3 Python Modules.\n${OPEN_PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON_DEPENDS)
    # TODO: handle dependencies with a loop over modules with MV-517
    list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-PYTHON_DEPENDS metavision-sdk-core-python metavision-sdk-core-ml-python)
endif (COMPILE_PYTHON3_BINDINGS)


######################################
#  metavision-openeb-python-samples  #
######################################
if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision OpenEB Python 3 libraries.\n${OPEN_PACKAGE_LICENSE}")
    # TODO: handle core-ml dependency inside loop with MV-517
    set(CPACK_COMPONENT_METAVISION-OPENEB-PYTHON-SAMPLES_DEPENDS metavision-hal-python-samples metavision-sdk-core-ml-python-samples)
    foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
        if(metavision-sdk-${available_open_module}-python-samples IN_LIST components_to_install_public)
            list(APPEND CPACK_COMPONENT_METAVISION-OPENEB-PYTHON-SAMPLES_DEPENDS metavision-sdk-${available_open_module}-python-samples)
        endif()
    endforeach(available_open_module)
endif (COMPILE_PYTHON3_BINDINGS)


###############################
#  metavision-openeb-license  #
###############################
set(CPACK_COMPONENT_METAVISION-OPENEB-LICENSE_DESCRIPTION "OpenEB License.\n${PACKAGE_LICENSE}")
# Add metavision-openeb-license to all open packages
foreach(available_open_module IN LISTS METAVISION_SDK_OPEN_MODULES_AVAILABLE)
      if(metavision-sdk-${available_open_module}-lib IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_open_module}-lib OPEN_MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${OPEN_MODULE_COMPONENT}_DEPENDS metavision-openeb-license)
      endif()
      if(metavision-sdk-${available_open_module}-python IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_open_module}-python OPEN_MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${OPEN_MODULE_COMPONENT}_DEPENDS metavision-openeb-license)
      endif()
      if(metavision-sdk-${available_open_module}-python-samples IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_open_module}-python-samples OPEN_MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${OPEN_MODULE_COMPONENT}_DEPENDS metavision-openeb-license)
      endif()
endforeach(available_open_module)


#######################
#  metavision-openeb  #
#######################
set(CPACK_COMPONENT_METAVISION-OPENEB_DESCRIPTION "OpenEB.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-OPENEB_DEPENDS metavision-openeb-bin metavision-openeb-lib metavision-openeb-dev metavision-openeb-samples)

if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-OPENEB_DEPENDS ${CPACK_COMPONENT_METAVISION-OPENEB_DEPENDS} metavision-openeb-python metavision-openeb-python-samples metavision-openeb-python${PYTHON3_DEFAULT_VERSION})
endif(COMPILE_PYTHON3_BINDINGS)

if (HDF5_FOUND)
    list(APPEND CPACK_DEBIAN_METAVISION-OPENEB_PACKAGE_DEPENDS "hdf5-plugin-ecf" "hdf5-plugin-ecf-dev")
endif (HDF5_FOUND)
string(REPLACE ";" ", " CPACK_DEBIAN_METAVISION-OPENEB_PACKAGE_DEPENDS "${CPACK_DEBIAN_METAVISION-OPENEB_PACKAGE_DEPENDS}")
