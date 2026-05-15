# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

include(CMakeParseArguments)

set_property(GLOBAL PROPERTY list_cpack_internal_components "")
set_property(GLOBAL PROPERTY list_cpack_public_components "")

#####################################################################
#
# Adds a cpack component
#
#
# usage :
#     add_cpack_component(
#         <PRIVATE|PUBLIC> <components>...
#        [<PRIVATE|PUBLIC> <components>...]
#      )
#
#
#  Adds the specified components to the list of debian packages to generate. Both PRIVATE and PUBLIC components will be
#  created when doing "cpack -G DEB", while only the PUBLIC component will be created when building target
#  "public_deb_packages"
#
#
function(add_cpack_component)

    cmake_parse_arguments(CPACK_COMPONENTS "" "" "PUBLIC;PRIVATE" ${ARGN})

    # Get current list of cpack components
    get_property(components_to_install_internal_tmp GLOBAL PROPERTY list_cpack_internal_components)
    get_property(components_to_install_public_tmp GLOBAL PROPERTY list_cpack_public_components)

    # Iterate through the input arguments to add them in the list of cpack components
    foreach(compon IN LISTS CPACK_COMPONENTS_PUBLIC)
        if (${compon} IN_LIST components_to_install_internal_tmp)
            message(SEND_ERROR
            "Error when calling function add_cpack_component : component ${compon} has already been listed as private, you cannot add it as public as well")
            return()
        endif()
        if (NOT ${compon} IN_LIST components_to_install_public_tmp)
            set (components_to_install_public_tmp ${components_to_install_public_tmp} ${compon})
        endif (NOT ${compon} IN_LIST components_to_install_public_tmp)
    endforeach(compon)
    foreach(compon IN LISTS CPACK_COMPONENTS_PRIVATE)
        if (${compon} IN_LIST components_to_install_public_tmp)
            message(SEND_ERROR
            "Error when calling function add_cpack_component : component ${compon} has already been listed as public, you cannot add it as private as well")
            return()
        endif()
        if (NOT ${compon} IN_LIST components_to_install_internal_tmp)
            set (components_to_install_internal_tmp ${components_to_install_internal_tmp} ${compon})
        endif (NOT ${compon} IN_LIST components_to_install_internal_tmp)
    endforeach(compon)

    # Set back global variables
    set_property(GLOBAL PROPERTY list_cpack_internal_components "${components_to_install_internal_tmp}")
    set_property(GLOBAL PROPERTY list_cpack_public_components "${components_to_install_public_tmp}")

endfunction(add_cpack_component)
