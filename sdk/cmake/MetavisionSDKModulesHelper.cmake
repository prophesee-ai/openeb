# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Create Metavision SDK Cmake Package

# Init variables
set(MetavisionSDK_CMAKE_OUTPUT_DIR "${GENERATE_FILES_DIRECTORY}/share/cmake/MetavisionSDKCMakePackagesFilesDir")
set(MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE share/cmake/MetavisionSDK)
set(MetavisionSDK_MODULES_CMAKE_CONFIG_SUBDIR Modules)
set(MetavisionSDK_COMPONENTS_CONFIG_INSTALL_PATH_RELATIVE
    ${MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE}/${MetavisionSDK_MODULES_CMAKE_CONFIG_SUBDIR})

###################################################
# Create a version file
# REMARK : if need version file during compilation, it will be best to create a tmp file and copy it only if different,
# to avoid re-compilation
set(VERSION_BARE_FILE version.h)
set(VERSION_FILE_INSTALLATION_PATH_RELATIVE include/metavision/sdk)
set(VERSION_FILE_OUTPUT_PATH ${GENERATE_FILES_DIRECTORY}/${VERSION_FILE_INSTALLATION_PATH_RELATIVE}/${VERSION_BARE_FILE})
add_library_version_header(generate_metavision_sdk_version_header
                           ${VERSION_FILE_OUTPUT_PATH}
                           metavision_sdk
)

###################################################
# Create Configuration file for the cmake package MetavisionSDK, that handles the components
include(CMakePackageConfigHelpers)

set(VERSION_FILE_INSTALLATION_PATH_RELATIVE include/metavision/sdk)
set(MetavisionSDK_config_file_to_install "${MetavisionSDK_CMAKE_OUTPUT_DIR}/MetavisionSDKConfig.cmake")
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/MetavisionSDKConfig.cmake.in"
    "${MetavisionSDK_config_file_to_install}"
    INSTALL_DESTINATION ${MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE}
)
set(MetavisionSDK_config_version_file_to_install "${MetavisionSDK_CMAKE_OUTPUT_DIR}/MetavisionSDKConfigVersion.cmake")
write_basic_package_version_file(
    "${MetavisionSDK_config_version_file_to_install}"
    COMPATIBILITY ExactVersion
)

install (FILES ${VERSION_FILE_OUTPUT_PATH}
         DESTINATION ${VERSION_FILE_INSTALLATION_PATH_RELATIVE}
         COMPONENT metavision-sdk-base-dev
         )

install(FILES "${MetavisionSDK_config_file_to_install}"
        DESTINATION ${MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE}
        COMPONENT metavision-sdk-base-dev
)

install(FILES "${MetavisionSDK_config_version_file_to_install}"
        DESTINATION ${MetavisionSDK_CMAKE_FILES_INSTALL_PATH_RELATIVE}
        COMPONENT metavision-sdk-base-dev
)

####################################################################
#
#
#
#
function(cmake_parse_arguments_only prefix parsedOptionalKeywords parsedSingleValueKeywords parsedMultiValueKeywords ignoredOptionalKeywords ignoredSingleValueKeywords ignoredMultiValueKeywords)

    set(optionalKeywords "")
    list(APPEND optionalKeywords "${parsedOptionalKeywords}")
    list(APPEND optionalKeywords "${ignoredOptionalKeywords}")
    set(singleValueKeywords "")
    list(APPEND singleValueKeywords "${parsedSingleValueKeywords}")
    list(APPEND singleValueKeywords "${ignoredSingleValueKeywords}")
    set(multiValueKeywords "")
    list(APPEND multiValueKeywords "${parsedMultiValueKeywords}")
    list(APPEND multiValueKeywords "${ignoredMultiValueKeywords}")

    cmake_parse_arguments(${prefix} "${optionalKeywords}" "${singleValueKeywords}" "${multiValueKeywords}" ${ARGN})

    set(IGNORED_ARGUMENTS "")
    foreach(kw IN LISTS ignoredOptionalKeywords)
        if(${prefix}_${kw})
            list(APPEND IGNORED_ARGUMENTS "${kw}")
        endif(${prefix}_${kw})
    endforeach()
    foreach(kw IN LISTS ignoredSingleValueKeywords)
        if(${prefix}_${kw})
            list(APPEND IGNORED_ARGUMENTS "${kw}")
            list(APPEND IGNORED_ARGUMENTS "${${prefix}_${kw}}")
        endif(${prefix}_${kw})
    endforeach()
    foreach(kw IN LISTS ignoredMultiValueKeywords)
        if(${prefix}_${kw})
            list(APPEND IGNORED_ARGUMENTS "${kw}")
            list(APPEND IGNORED_ARGUMENTS "${${prefix}_${kw}}")
        endif(${prefix}_${kw})
    endforeach()
    set(${prefix}_IGNORED_ARGUMENTS "${IGNORED_ARGUMENTS}" PARENT_SCOPE)

    set(parsedKeywords "")
    list(APPEND parsedKeywords "${parsedOptionalKeywords}")
    list(APPEND parsedKeywords "${parsedSingleValueKeywords}")
    list(APPEND parsedKeywords "${parsedMultiValueKeywords}")
    foreach(kw IN LISTS parsedKeywords)
        set(${prefix}_${kw} "${${prefix}_${kw}}" PARENT_SCOPE)
    endforeach()

endfunction(cmake_parse_arguments_only)

#####################################################################
#
# Creates Metavision SDK module library, installs it and adds a cmake component for package MetavisionSDK
#
#
# usage :
#     MetavisionSDK_add_module(<module-name>
#          [INTERFACE | SOURCES srcs...]
#          [[REQUIRED_METAVISION_SDK_MODULES
#               <PRIVATE|PUBLIC|INTERFACE> <components>...
#               [<PRIVATE|PUBLIC|INTERFACE> <components>...]
#          ]]
#          [EXTRA_REQUIRED_PACKAGE packages...]
#      )
#
#
#  Creates and installs a library named metavision_sdk_<module-name> . This library's public headers need to be in a
#  folder "include" at sdk/modules/<module-name>. This function handles the right cpack components (i.e.
#  metavision-sdk-<module-name> for runtime and metavision-sdk-<module-name>-dev for development). Moreover, it creates
#  (and installs) the files needed to create a <module-name> module of the MetavisionSDK package. Finally, an alias
#  library MetavisionSDK::<module-name> will also be created
#
#  REMARK : given the above description, note that folder sdk/modules/<module-name>/include has to exist
#
#  The INTERFACE is to be used if the library metavision_sdk_<module-name> is to be a INTERFACE library. It will be
#  a SHARED one otherwise. In this second case, you may use the option SOURCES to specify the library sources. However,
#  this is not mandatory, as sources may be specified with target_sources() after the call to this function.
#
#  The export set where the library metavision_sdk_<module-name> is exported will be named
#  metavision_sdk_<module-name>Targets. This means that if you want to add other targets to the set, you'll have to use
#  this name, as it is the one installed in this function.
#
#  The REQUIRED_METAVISION_SDK_MODULES defines a list of other MetavisionSDK components on which the current one
#  depends on. If this option is provided :
#     -> a call to target_link_libraries(metavision_sdk_<module-name> <PRIVATE|PUBLIC|INTERFACE> ..) will be done
#        (note that if metavision_sdk_<module-name> is an interface library, only the INTERFACE keyword is allowed)
#     -> for INTERFACE and PUBLIC, an additional file MetavisionSDK_<module-name>Depends.cmake will be created. This
#        file is used by the MetavisionSDK find package to determine inter-dependencies between the different cmake
#        components
#
#  The EXTRA_REQUIRED_PACKAGE defines a list of other packages the module depends on. If you need only some components
#  of these packages, you need to provide them as a full string. For example "Boost COMPONENTS filesystem" if the
#  module only needs the filesystem component of Boost. The list provided with this option is used to create the
#  configuration file MetavisionSDK_<module-name>Config.cmake, used when looking for package MetavisionSDK.
#  REMARK : please note that when using this option, you'll still need to call
#                     target_link_libraries(metavision_sdk_<module-name> ...)
#           with the target the package defines.
#
#
#  Example :
#
#  MetavisionSDK_add_module(analytics
#      REQUIRED_METAVISION_SDK_MODULES
#          PUBLIC core
#          PRIVATE cv
#      EXTRA_REQUIRED_PACKAGE "Boost COMPONENTS system filesystem"
#      EXTRA_REQUIRED_PACKAGE OpenCV
#  )
#
#  will generate the following code :
#
#
#      add_library(metavision_sdk_analytics SHARED)
#      add_library(MetavisionSDK::analytics ALIAS metavision_sdk_analytics)
#
#
#      set_target_properties(metavision_sdk_analytics
#          PROPERTIES
#              VERSION ${PROJECT_VERSION_FULL}
#              SOVERSION ${PROJECT_VERSION_MAJOR}
#      )
#
#      target_include_directories(metavision_sdk_analytics
#          PUBLIC
#              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sdk/modules/analytics/include>
#              $<INSTALL_INTERFACE:include>
#          PRIVATE
#              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sdk/modules/analytics/src/include> # only if folder exists
#
#      )
#
#      target_link_libraries(metavision_sdk_analytics
#          PUBLIC
#              MetavisionSDK::base
#          PRIVATE
#              MetavisionSDK::cv
#      )
#
#      install(TARGETS metavision_sdk_analytics
#          EXPORT metavision_sdk_analyticsTargets
#          RUNTIME
#              DESTINATION bin
#              COMPONENT metavision-sdk-analytics-lib
#          ARCHIVE
#              DESTINATION lib
#              COMPONENT metavision-sdk-analytics-lib
#          LIBRARY
#              DESTINATION lib
#              COMPONENT metavision-sdk-analytics-lib
#              NAMELINK_SKIP
#      )
#
#
#      install(TARGETS metavision_sdk_analytics
#          EXPORT metavision_sdk_analyticsTargets
#          LIBRARY
#              DESTINATION lib
#              COMPONENT metavision-sdk-analytics-dev
#              NAMELINK_ONLY
#       )
#
#       # Install public headers
#       install(DIRECTORY ${PROJECT_SOURCE_DIR}/sdk/modules/analytics/include/
#               DESTINATION include
#               COMPONENT metavision-sdk-analytics-dev
#       )
#
#
#  and it will generate the needed cmake files to add a component "analytics" to MetavisionSDK cmake package. This
#  component will depend on the core one, so that when doing
#
#       find_package(MetavisionSDK COMPONENTS analytics)
#
#  the core module will be automatically looked for. Finally, in the configuration file generated the packages Boost
#  will also be looked for.
#
#
include(CMakeParseArguments)
function(MetavisionSDK_add_module module_name)
    set(multiValueArgs SOURCES EXTRA_REQUIRED_PACKAGE REQUIRED_METAVISION_SDK_MODULES VARIABLE_TO_SET VARIABLE_TO_SET_WIN32)
    cmake_parse_arguments(PARSED_ARGS "INTERFACE_LIBRARY" "" "${multiValueArgs}" ${ARGN})

    # Check validity of input args
    if(PARSED_ARGS_SOURCES AND PARSED_ARGS_INTERFACE_LIBRARY)
        message(SEND_ERROR
        "Error when calling function MetavisionSDK_add_module : incompatible input arguments INTERFACE_LIBRARY and SOURCES")
        return()
    endif(PARSED_ARGS_SOURCES AND PARSED_ARGS_INTERFACE_LIBRARY)

    set (module_include_folder_path ${PROJECT_SOURCE_DIR}/sdk/modules/${module_name}/include)
    if (NOT IS_DIRECTORY "${module_include_folder_path}")
        set (module_include_folder_path ${PROJECT_SOURCE_DIR}/sdk/modules/${module_name}/cpp/include)
        if( NOT IS_DIRECTORY "${module_include_folder_path}")
            message(SEND_ERROR
                    "Error in MetavisionSDK_add_module : folder ${module_include_folder_path} does not exist")
            return()
        endif  (NOT IS_DIRECTORY "${module_include_folder_path}")
    endif (NOT IS_DIRECTORY "${module_include_folder_path}")


    # Create library:
    set(public_keywork_to_use PUBLIC)
    if(PARSED_ARGS_INTERFACE_LIBRARY)
        add_library(metavision_sdk_${module_name} INTERFACE)
        set(public_keywork_to_use INTERFACE)
    else()
        add_library(metavision_sdk_${module_name} SHARED ${PARSED_ARGS_SOURCES})
        set_target_properties(metavision_sdk_${module_name}
            PROPERTIES
                VERSION ${PROJECT_VERSION_FULL}
                SOVERSION ${PROJECT_VERSION_MAJOR}
        )
    endif(PARSED_ARGS_INTERFACE_LIBRARY)

    # Create library alias:
    add_library(MetavisionSDK::${module_name} ALIAS metavision_sdk_${module_name})

    # Set export name of the built target, so that we can refer to it with MetavisionSDK::${module_name}, thanks
    # to the fact that we export the target set with the NAMESPACE MetavisionSDK:: (see below when installing export)
    set_target_properties(metavision_sdk_${module_name} PROPERTIES EXPORT_NAME ${module_name})

    target_include_directories(metavision_sdk_${module_name}
        ${public_keywork_to_use}
            $<BUILD_INTERFACE:${module_include_folder_path}>
            $<INSTALL_INTERFACE:include>
    )

    set(internal_include_dir "${PROJECT_SOURCE_DIR}/sdk/modules/${module_name}/cpp/src/include")
    if(NOT PARSED_ARGS_INTERFACE_LIBRARY)
        if(IS_DIRECTORY "${internal_include_dir}")
            target_include_directories(metavision_sdk_${module_name}
                PRIVATE
                    $<BUILD_INTERFACE:${internal_include_dir}>
            )
        endif(IS_DIRECTORY "${internal_include_dir}")
    endif(NOT PARSED_ARGS_INTERFACE_LIBRARY)

    set(needed_metavision_sdk_cmake_components)
    if (PARSED_ARGS_REQUIRED_METAVISION_SDK_MODULES)
        cmake_parse_arguments(SDK_NEEDED_MODULES "" "" "PUBLIC;INTERFACE;PRIVATE" ${PARSED_ARGS_REQUIRED_METAVISION_SDK_MODULES})
        foreach(mod IN LISTS SDK_NEEDED_MODULES_PUBLIC)
            target_link_libraries(metavision_sdk_${module_name}
                PUBLIC MetavisionSDK::${mod}
            )
            list(APPEND needed_metavision_sdk_cmake_components ${mod})
        endforeach(mod)

        foreach(mod IN LISTS SDK_NEEDED_MODULES_INTERFACE)
            target_link_libraries(metavision_sdk_${module_name}
                INTERFACE MetavisionSDK::${mod}
            )
            list(APPEND needed_metavision_sdk_cmake_components ${mod})
        endforeach(mod)

        foreach(mod IN LISTS SDK_NEEDED_MODULES_PRIVATE)
            target_link_libraries(metavision_sdk_${module_name}
                PRIVATE MetavisionSDK::${mod}
            )
        endforeach(mod)
    endif (PARSED_ARGS_REQUIRED_METAVISION_SDK_MODULES)


    # Install target :
    set(COMPONENT_NAME_PREFIX "metavision-sdk-${module_name}")
    string(REPLACE "_" "-" COMPONENT_NAME_PREFIX "${COMPONENT_NAME_PREFIX}")
    if(PARSED_ARGS_INTERFACE_LIBRARY)
        install(TARGETS metavision_sdk_${module_name}
                EXPORT metavision_sdk_${module_name}Targets
        )
    else()
        install(TARGETS metavision_sdk_${module_name}
            EXPORT metavision_sdk_${module_name}Targets
            RUNTIME
                DESTINATION ${RUNTIME_INSTALL_DEST}
                COMPONENT ${COMPONENT_NAME_PREFIX}-lib
            ARCHIVE
                DESTINATION ${ARCHIVE_INSTALL_DEST}
                COMPONENT ${COMPONENT_NAME_PREFIX}-lib
            LIBRARY
                DESTINATION ${LIBRARY_INSTALL_DEST}
                COMPONENT ${COMPONENT_NAME_PREFIX}-lib
                NAMELINK_SKIP
            )

        install(TARGETS metavision_sdk_${module_name}
                EXPORT metavision_sdk_${module_name}Targets
                LIBRARY
                    DESTINATION ${LIBRARY_INSTALL_DEST}
                    COMPONENT ${COMPONENT_NAME_PREFIX}-dev
                    NAMELINK_ONLY
            )
    endif(PARSED_ARGS_INTERFACE_LIBRARY)

    # Install public headers
    install(DIRECTORY ${module_include_folder_path}/
            DESTINATION ${HEADER_INSTALL_DEST}
            COMPONENT ${COMPONENT_NAME_PREFIX}-dev
            )

    # Create the configuration file
    set(config_file_text "")
    set(config_file_text "${config_file_text}if (ANDROID)\n")
    set(config_file_text "${config_file_text}    include(\${CMAKE_CURRENT_LIST_DIR}/../../android/env.cmake)\n")
    set(config_file_text "${config_file_text}endif (ANDROID)\n\n")

    foreach(variable_to_set ${PARSED_ARGS_VARIABLE_TO_SET})
        set(config_file_text "${config_file_text}set(${variable_to_set})\n")
    endforeach(variable_to_set)

    if(PARSED_ARGS_VARIABLE_TO_SET_WIN32)
    string(APPEND config_file_text "if(WIN32)\n")
      foreach(win32_variable_to_set ${PARSED_ARGS_VARIABLE_TO_SET_WIN32})
        string(APPEND config_file_text "  set(${win32_variable_to_set})\n")
      endforeach(win32_variable_to_set)
      string(APPEND config_file_text "endif(WIN32)\n\n")
    endif(PARSED_ARGS_VARIABLE_TO_SET_WIN32)

    foreach(extra_package ${PARSED_ARGS_EXTRA_REQUIRED_PACKAGE})
        set(config_file_text "${config_file_text}find_package(${extra_package} REQUIRED QUIET)\n")
    endforeach(extra_package)

    set(config_file_text "${config_file_text}\nif(NOT TARGET MetavisionSDK::${module_name})")
    set(config_file_text
        "${config_file_text}\n    include(\"\${CMAKE_CURRENT_LIST_DIR}/MetavisionSDK_${module_name}Targets.cmake\")")
    set(config_file_text "${config_file_text}\nendif(NOT TARGET MetavisionSDK::${module_name})")

    # Write the config file
    set(input_conf_path "${MetavisionSDK_CMAKE_OUTPUT_DIR}/MetavisionSDK_${module_name}Config.cmake.in")
    file(WRITE ${input_conf_path} "${config_file_text}")


    # Get installation path
    set(MVPackageModule_CMAKE_FILES_INSTALLATION_PATH_RELATIVE
        "${MetavisionSDK_COMPONENTS_CONFIG_INSTALL_PATH_RELATIVE}/${module_name}")
    set(MVPackageModule_CMAKE_PACKAGE_OUTPUT_FILES_DIR
        "${MetavisionSDK_CMAKE_OUTPUT_DIR}/${MetavisionSDK_MODULES_CMAKE_CONFIG_SUBDIR}/${module_name}")
    include(CMakePackageConfigHelpers)

    set(files_to_install "")
    set(output_config_file_path
        "${MVPackageModule_CMAKE_PACKAGE_OUTPUT_FILES_DIR}/MetavisionSDK_${module_name}Config.cmake")
    configure_package_config_file(
       "${input_conf_path}"
        "${output_config_file_path}"
        INSTALL_DESTINATION ${MVPackageModule_CMAKE_FILES_INSTALLATION_PATH_RELATIVE}
    )
    list(APPEND files_to_install "${output_config_file_path}")

    set(output_config_version_file_path
        "${MVPackageModule_CMAKE_PACKAGE_OUTPUT_FILES_DIR}/MetavisionSDK_${module_name}ConfigVersion.cmake")
    write_basic_package_version_file(
        "${output_config_version_file_path}"
        COMPATIBILITY ExactVersion
    )
    list(APPEND files_to_install "${output_config_version_file_path}")

    if(needed_metavision_sdk_cmake_components)
        # Create a depends file
        set(output_depends_file_path
            "${MVPackageModule_CMAKE_PACKAGE_OUTPUT_FILES_DIR}/MetavisionSDK_${module_name}Depends.cmake")
        file(WRITE ${output_depends_file_path}
            "set(MetavisionSDK_${module_name}_NEEDED_COMPONENTS ${needed_metavision_sdk_cmake_components})")
        list(APPEND files_to_install "${output_depends_file_path}")
    endif()

    export(EXPORT metavision_sdk_${module_name}Targets
           FILE "${MetavisionSDK_CMAKE_OUTPUT_DIR}/${MetavisionSDK_MODULES_CMAKE_CONFIG_SUBDIR}/${module_name}/MetavisionSDK_${module_name}Targets.cmake"
           NAMESPACE MetavisionSDK::
    )

    install(EXPORT metavision_sdk_${module_name}Targets
            FILE MetavisionSDK_${module_name}Targets.cmake
            NAMESPACE MetavisionSDK::
            DESTINATION ${MVPackageModule_CMAKE_FILES_INSTALLATION_PATH_RELATIVE}
            COMPONENT ${COMPONENT_NAME_PREFIX}-dev
    )


    install(FILES ${files_to_install}
            DESTINATION ${MVPackageModule_CMAKE_FILES_INSTALLATION_PATH_RELATIVE}
            COMPONENT ${COMPONENT_NAME_PREFIX}-dev
    )

endfunction(MetavisionSDK_add_module)


#####################################################################
#
# Forward call to MetavisionSDK_add_module and install sdk advanced files
#
#
function(MetavisionSDK_add_advanced_module module_name)

    MetavisionSDK_add_module(${module_name} ${ARGN})

    if(EXISTS "${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK")
        install(FILES
                    ${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK
                    ${PROJECT_SOURCE_DIR}/licensing/OPEN_SOURCE_3RDPARTY_NOTICES
                DESTINATION share/metavision/licensing)
    endif()

endfunction(MetavisionSDK_add_advanced_module)
