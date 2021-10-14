# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

# Get the list of the registered public components
get_property(components_to_install_public GLOBAL PROPERTY list_cpack_public_components)

#############################
#  metavision-sdk-pro-lib   #
#############################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-SDK-PRO-LIB_DESCRIPTION "Metavision SDK Professional libraries.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-PRO-LIB_DEPENDS metavision-open-lib)
foreach(available_professional_module IN LISTS METAVISION_SDK_PROFESSIONAL_MODULES_AVAILABLE)
    if(metavision-sdk-${available_professional_module}-lib IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-PRO-LIB_DEPENDS metavision-sdk-${available_professional_module}-lib)
    endif()
endforeach(available_professional_module)

#############################
#  metavision-sdk-pro-bin   #
#############################
set(CPACK_COMPONENT_METAVISION-SDK-PRO-BIN_DESCRIPTION "Metavision SDK Professional applications.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-PRO-BIN_DEPENDS metavision-open-bin)
foreach(available_professional_module IN LISTS METAVISION_SDK_PROFESSIONAL_MODULES_AVAILABLE)
    if(metavision-sdk-${available_professional_module}-bin IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-PRO-BIN_DEPENDS metavision-sdk-${available_professional_module}-bin)
    endif()
endforeach(available_professional_module)

################################
# metavision-sdk-pro-python3.X #
################################
foreach (py_suffix ${PYTHON3_ALL_VERSIONS})
    set(CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON${py_suffix}_DESCRIPTION "Metavision SDK Professional Python 3 libraries.\n${PACKAGE_LICENSE}")
    set(CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON${py_suffix}_DEPENDS metavision-open-python${py_suffix})
    foreach(available_professional_module IN LISTS METAVISION_SDK_PROFESSIONAL_MODULES_AVAILABLE)
        if(metavision-sdk-${available_professional_module}-python${py_suffix} IN_LIST components_to_install_public)
            list(APPEND CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON${py_suffix}_DEPENDS metavision-sdk-${available_professional_module}-python${py_suffix})
        endif()
    endforeach(available_professional_module)
endforeach()

#####################################
# metavision-sdk-pro-python-samples #
#####################################
set(CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision SDK Professional Python 3 libraries.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON-SAMPLES_DEPENDS metavision-open-python-samples)
foreach(available_professional_module IN LISTS METAVISION_SDK_PROFESSIONAL_MODULES_AVAILABLE)
    if(metavision-sdk-${available_professional_module}-python-samples IN_LIST components_to_install_public)
        list(APPEND CPACK_COMPONENT_METAVISION-SDK-PRO-PYTHON-SAMPLES_DEPENDS metavision-sdk-${available_professional_module}-python-samples)
    endif()
endforeach(available_professional_module)


