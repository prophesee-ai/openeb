# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set (PACKAGE_LICENSE "License: Copyright (c) Prophesee S.A. - All Rights Reserved")
set (OPEN_PACKAGE_LICENSE "License : Copyright (c) Prophesee S.A.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "")

set(CPACK_PACKAGE_VENDOR "Prophesee")

set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_FULL})

set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar") # Variable introduced in cmake 3.7 (see https://cmake.org/cmake/help/v3.8/release/3.7.html#cpack)

# Follow Debian policy on control files' permissions
# cf https://cmake.org/cmake/help/v3.10/module/CPackDeb.html#variable:CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)

set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "support@prophesee.ai") #required
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE https://support.prophesee.ai)

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

# ! ATTENTION : the following variables work only for cpack version >=3.6, otherwise they are ignored !!
#
# To install cmake (and cpack) version 3.10.2 :
# cd $HOME
# wget https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.sh
# mkdir /opt/cmake-3.10.2
# sudo sh cmake-3.10.2-Linux-x86_64.sh --prefix=/opt/cmake-3.10.2/ --exclude-subdir
#
#
# To generate the packages, once done cmake (the default one is ok, you do not need to
# use the one installed with the command above) and make, in your build directory do :
#
# sudo  /opt/cmake-3.10.2/bin/cpack -G DEB


# CPACK_DEBIAN_<component>_FILE_NAME = name of the file built by cpack (ex myPack-0.2.deb)
# CPACK_DEBIAN_<component>_PACKAGE_NAME = name of the installed package
# CPACK_COMPONENT_<component>_DESCRIPTION = package description
# CPACK_COMPONENT_<component>_DEPENDS (optional) = other components the package may depend on.
#
# REMARK : replace <component> for the UPPER CASE component name

###########################
#        Metavision       #
###########################

# Include cpack configuration from the public packages of MetavisionSDK
include(${PROJECT_SOURCE_DIR}/sdk/cmake/MetavisionOffersCPackConfig.cmake)

###########################
#           HAL           #
###########################

# Include cpack configuration from the public packages of MetavisionHAL
include(${PROJECT_SOURCE_DIR}/hal/cmake/MetavisionHALCPackConfig.cmake)
if(EXISTS "${PROJECT_SOURCE_DIR}/hal_psee_plugins/cmake/MetavisionHALPseePluginsCPackConfig.cmake")
    include("${PROJECT_SOURCE_DIR}/hal_psee_plugins/cmake/MetavisionHALPseePluginsCPackConfig.cmake")
endif(EXISTS "${PROJECT_SOURCE_DIR}/hal_psee_plugins/cmake/MetavisionHALPseePluginsCPackConfig.cmake")

################################
#      Standalone samples     ##
################################
set(CPACK_COMPONENT_METAVISION-DECODERS-SAMPLES_DESCRIPTION "Metavision samples on how to decode raw data\n${OPEN_PACKAGE_LICENSE}")

# Remove local-install components, these are not meant to be used with cpack
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
foreach(comp ${CPACK_COMPONENTS_ALL})
    if (${comp} MATCHES "-local-install$")
        list(REMOVE_ITEM CPACK_COMPONENTS_ALL ${comp})
    endif()
endforeach()

include(CPack) # This has to be at the end