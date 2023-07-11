# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

get_property(cpack_public_components GLOBAL PROPERTY list_cpack_public_components)
get_property(cpack_internal_components GLOBAL PROPERTY list_cpack_internal_components)

set(cpack_all_components ${cpack_public_components} ${cpack_internal_components})

foreach(comp ${cpack_all_components})
    string(TOUPPER ${comp} COMP_UPPER)
    string(TOLOWER ${comp} comp_lower)

    # File name
    if (NOT DEFINED CPACK_DEBIAN_${COMP_UPPER}_FILE_NAME)
        set(CPACK_DEBIAN_${COMP_UPPER}_FILE_NAME "${comp_lower}-${PROJECT_VERSION_FULL}.deb")
    endif (NOT DEFINED CPACK_DEBIAN_${COMP_UPPER}_FILE_NAME)

    # package installed name
    if (NOT DEFINED CPACK_DEBIAN_${COMP_UPPER}_PACKAGE_NAME)
        set(CPACK_DEBIAN_${COMP_UPPER}_PACKAGE_NAME "${comp_lower}")
    endif (NOT DEFINED CPACK_DEBIAN_${COMP_UPPER}_PACKAGE_NAME)
endforeach(comp)

# Need to include the following after having set the variables above
include(MetavisionCPackConfig)

foreach(PackageGroup All Public)
    string(TOLOWER ${PackageGroup} package_group_lower)

    if (cpack_${package_group_lower}_components)
       list(SORT cpack_${package_group_lower}_components) # sort the list, so that it's easier to see the diff
        set(PACKAGES_CONFIG_FILE ${GENERATE_FILES_DIRECTORY}/CPackConfig${PackageGroup}.cmake)
        file(COPY ${CMAKE_BINARY_DIR}/CPackConfig.cmake DESTINATION ${GENERATE_FILES_DIRECTORY})
        file(RENAME ${GENERATE_FILES_DIRECTORY}/CPackConfig.cmake ${PACKAGES_CONFIG_FILE})
        file(APPEND ${PACKAGES_CONFIG_FILE} "SET(CPACK_COMPONENTS_ALL ${cpack_${package_group_lower}_components})\n")
        file(APPEND ${PACKAGES_CONFIG_FILE} "SET(CPACK_OUTPUT_FILE_PREFIX packages/${package_group_lower})")

        add_custom_target(${package_group_lower}_build
                          COMMAND ${CUSTOM_COMMAND_RECURSIVE_MAKEFILE_TOKEN} ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} # To make sure that project has been built (otherwise cpack would wait for a long time - user may think that is blocking)
                          COMMENT "Building project. Please wait..."
        )
        add_custom_target(${package_group_lower}_deb_packages
                          COMMAND LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_CPACK_COMMAND} -G DEB --config ${PACKAGES_CONFIG_FILE} -V
                          COMMENT "Running CPack. Please wait..."
        )

        # setting proper dependencies : 
        #                       |-- <all/public>_deb1 <--|
        # <all/public>_build <--|--        ...        <--|-- <all/public>_deb_packages 
        #                       |-- <all/public>_debN <--|
        add_dependencies(${package_group_lower}_deb_packages ${package_group_lower}_build)
        foreach (dep ${${package_group_lower}_deb_packages_dependencies})
            if(TARGET ${dep})
                add_dependencies(${dep} ${package_group_lower}_build)
                add_dependencies(${package_group_lower}_deb_packages ${dep})
            endif()
        endforeach (dep ${${package_group_lower}_deb_packages_dependencies})
    else()
        add_custom_target(${package_group_lower}_deb_packages
                          COMMAND ${CMAKE_COMMAND} -E "WARNING : no ${package_group_lower} packages registered"
        )
    endif(cpack_${package_group_lower}_components)

endforeach(PackageGroup)
