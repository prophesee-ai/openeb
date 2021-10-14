# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

####################################
# metavision-hal-openeb-plugins #
####################################

# File and package name of the components are automatically set, just need to set the package description
set(CPACK_COMPONENT_METAVISION-HAL-OPENEB-PLUGINS_DESCRIPTION "OpenEB Plugins for Metavision HAL.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-HAL-OPENEB-PLUGINS_DEPENDS metavision-hal-lib metavision-sdk-base-lib)
