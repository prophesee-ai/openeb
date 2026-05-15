# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

# Get the list of the registered public components
get_property(components_to_install_public GLOBAL PROPERTY list_cpack_public_components)

##################################
#  metavision-sdk-advanced-lib   #
##################################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-LIB_DESCRIPTION "Metavision SDK Advanced libraries.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-LIB_DEPENDS metavision-openeb-lib)
foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
    if(metavision-sdk-${available_module}-lib IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-LIB_DEPENDS metavision-sdk-${available_module}-lib)
    endif()
endforeach(available_module)

##################################
#  metavision-sdk-advanced-bin   #
##################################

set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-BIN_DESCRIPTION "Metavision SDK Advanced applications.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-BIN_DEPENDS metavision-sdk-advanced-lib)
foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
    if(metavision-sdk-${available_module}-bin IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-BIN_DEPENDS metavision-sdk-${available_module}-bin)
    endif()
endforeach(available_module)

####################################
#   metavision-sdk-advanced-dev    #
####################################
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-DEV_DESCRIPTION "Development (C++) files for Metavision SDK Advanced libraries.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-DEV_DEPENDS metavision-sdk-advanced-lib)
foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
    if(metavision-sdk-${available_module}-dev IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-DEV_DEPENDS metavision-sdk-${available_module}-dev)
    endif()
endforeach(available_module)

#####################################
#   metavision-sdk-advanced-samples #
#####################################
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-SAMPLES_DESCRIPTION "Samples for Metavision SDK Advanced libraries.\n${OPEN_PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-SAMPLES_DEPENDS metavision-sdk-advanced-dev)
foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
    if(metavision-sdk-${available_module}-samples IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-SAMPLES_DEPENDS metavision-sdk-${available_module}-samples)
    endif()
endforeach(available_module)

#####################################
# metavision-sdk-advanced-python3.X #
#####################################
if (COMPILE_PYTHON3_BINDINGS)
    foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
        set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON${py_suffix}_DESCRIPTION "Metavision SDK Advanced Python 3 libraries.\n${PACKAGE_LICENSE}")
        set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON${py_suffix}_DEPENDS metavision-openeb-python${py_suffix} metavision-sdk-advanced-lib)
        foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
            if(metavision-sdk-${available_module}-python${py_suffix} IN_LIST components_to_install_public)
                list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON${py_suffix}_DEPENDS metavision-sdk-${available_module}-python${py_suffix})
            endif()
        endforeach(available_module)
    endforeach()
endif(COMPILE_PYTHON3_BINDINGS)

##################################
# metavision-sdk-advanced-python #
##################################
if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON_DESCRIPTION "Metavision SDK Advanced Python 3 Modules.\n${PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON_DEPENDS metavision-openeb-python)
    foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
        if(metavision-sdk-${available_module}-python IN_LIST components_to_install_public)
            list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON_DEPENDS metavision-sdk-${available_module}-python)
        endif()
    endforeach(available_module)
endif(COMPILE_PYTHON3_BINDINGS)

##########################################
# metavision-sdk-advanced-python-samples #
##########################################
if (COMPILE_PYTHON3_BINDINGS)
    set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision SDK Advanced Python 3 libraries.\n${PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON-SAMPLES_DEPENDS)
    foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
        if(metavision-sdk-${available_module}-python-samples IN_LIST components_to_install_public)
          list(APPEND CPACK_COMPONENT_METAVISION-SDK-ADVANCED-PYTHON-SAMPLES_DEPENDS metavision-sdk-${available_module}-python-samples)
        endif()
    endforeach(available_module)
endif(COMPILE_PYTHON3_BINDINGS)

#####################################
#  metavision-sdk-advanced-license  #
#####################################
set(CPACK_COMPONENT_METAVISION-SDK-ADVANCED-LICENSE_DESCRIPTION "Metavision SDK License.\n${PACKAGE_LICENSE}")
# Add metavision-sdk-advanced to all sdk advanced packages
foreach(available_module IN LISTS METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
     if(metavision-sdk-${available_module}-lib IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_module}-lib MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${MODULE_COMPONENT}_DEPENDS metavision-sdk-advanced-license)
      endif()
      if(metavision-sdk-${available_module}-python IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_module}-python MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${MODULE_COMPONENT}_DEPENDS metavision-sdk-advanced-license)
      endif()
      if(metavision-sdk-${available_module}-python-samples IN_LIST components_to_install_public)
        string(TOUPPER metavision-sdk-${available_module}-python-samples MODULE_COMPONENT)
        list(APPEND CPACK_COMPONENT_${MODULE_COMPONENT}_DEPENDS metavision-sdk-advanced-license)
      endif()
endforeach(available_module)
