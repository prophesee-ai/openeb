# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Compile code that is used multiple times in object libraries :

set(lib_common_obj_list metavision_hal_psee_plugin_obj)

add_library(metavision_psee_hw_layer_obj OBJECT)
target_link_libraries(metavision_psee_hw_layer_obj
    PUBLIC
        metavision_hal
)

# The following line is needed because when linking a shared library to an object one,
# the object library needs to be compiled with -fPIC
# cf https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
set_target_properties(metavision_psee_hw_layer_obj
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(metavision_psee_hw_layer_obj
    PUBLIC
        libusb-1.0
)

target_include_directories(metavision_psee_hw_layer_obj
    PUBLIC
        ${metavision_psee_hw_layer_include_dir}
    PRIVATE
        ${biasgen_calib_files_dir}
        ${metavision_hal_psee_plugins_include_dir}
)

add_library(metavision_psee_hw_layer SHARED)

target_sources(metavision_psee_hw_layer
    PRIVATE
        $<TARGET_OBJECTS:metavision_psee_hw_layer_obj>
)

target_include_directories(metavision_psee_hw_layer
  PUBLIC
     $<BUILD_INTERFACE:${metavision_psee_hw_layer_include_dir}>
     $<INSTALL_INTERFACE:include>
)

target_link_libraries(metavision_psee_hw_layer
    PRIVATE
        metavision_psee_hw_layer_obj
    PUBLIC
        metavision_hal
)

install(TARGETS metavision_psee_hw_layer
    EXPORT MetavisionPSEEHWLayerTargets
    RUNTIME
        DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}"
        COMPONENT metavision-hal-prophesee-hw-layer-lib
    ARCHIVE
        DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}"
        COMPONENT metavision-hal-prophesee-hw-layer-lib
    LIBRARY
        DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}"
        COMPONENT metavision-hal-prophesee-hw-layer-lib
)

add_custom_command(TARGET metavision_psee_hw_layer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_BUILD_PLUGIN_PATH}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:metavision_psee_hw_layer>" "${HAL_BUILD_PLUGIN_PATH}")

if(HAL_SENSORLIB_SUPPORT)
    add_custom_command(TARGET metavision_psee_hw_layer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_BUILD_PLUGIN_PATH}/sensorlib"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:metavision_psee_hw_layer>" "${HAL_BUILD_PLUGIN_PATH}/sensorlib")
endif(HAL_SENSORLIB_SUPPORT)

# Install public headers
install(DIRECTORY ${metavision_psee_hw_layer_include_dir}/metavision
    DESTINATION include
    COMPONENT metavision-hal-prophesee-hw-layer-dev
)

set(METAVISION_PSEEHWLayer_EXPORT_FILE_NAME MetavisionPSEEHWLayerTargets.cmake)
set(metavision_psee_hw_layer_config_files_output_dir "${GENERATE_FILES_DIRECTORY}/share/cmake/MetavisionPSEEHWLayerCMakePackagesFilesDir")
add_library(Metavision::PSEEHWLayer ALIAS metavision_psee_hw_layer)

set_target_properties(metavision_psee_hw_layer
    PROPERTIES
        EXPORT_NAME PSEEHWLayer
)

export(EXPORT MetavisionPSEEHWLayerTargets
       FILE ${metavision_psee_hw_layer_config_files_output_dir}/${METAVISION_PSEEHWLayer_EXPORT_FILE_NAME}
       NAMESPACE Metavision::
)

install(EXPORT MetavisionPSEEHWLayerTargets
        FILE ${METAVISION_PSEEHWLayer_EXPORT_FILE_NAME}
        NAMESPACE Metavision::
        DESTINATION share/cmake/MetavisionPSEEHWLayer
        COMPONENT metavision-hal-prophesee-hw-layer-dev
)

# Create and install configuration files
include(CMakePackageConfigHelpers)
set(metavision_psee_hw_layer_config_file "${metavision_psee_hw_layer_config_files_output_dir}/MetavisionPSEEHWLayerConfig.cmake")
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/hal_psee_plugins/cmake/MetavisionPSEEHWLayerConfig.cmake.in"
    "${metavision_psee_hw_layer_config_file}"
    INSTALL_DESTINATION share/cmake/MetavisionPSEEHWLayer
)
set(metavision_psee_hw_layer_config_version_file "${metavision_psee_hw_layer_config_files_output_dir}/MetavisionPSEEHWLayerConfigVersion.cmake")
write_basic_package_version_file(
    "${metavision_psee_hw_layer_config_version_file}"
    COMPATIBILITY ExactVersion
)

install(FILES ${metavision_psee_hw_layer_config_file} ${metavision_psee_hw_layer_config_version_file}
        DESTINATION share/cmake/MetavisionPSEEHWLayer
        COMPONENT metavision-hal-prophesee-hw-layer-dev
)

set(lib_camera_obj_list "")

set(lib_camera_obj_list
        metavision_hal_gen3_pattern_checkers_obj
        metavision_hal_gen31_camera_obj
        metavision_hal_fx3_protocol_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_treuzell_unknown_obj
        metavision_hal_psee_video_obj
        metavision_hal_ti_tmp103_obj
        metavision_hal_gen31_evk2_system_obj
        metavision_hal_gen31_evk3_system_obj
        metavision_hal_gen41_evk2_system_obj
        metavision_hal_gen41_evk3_system_obj
        metavision_hal_imx636_evk2_system_obj
        metavision_hal_imx636_evk3_system_obj
        metavision_hal_imx646_evk3_system_obj
        metavision_hal_imx636_rdk2_system_obj
        metavision_hal_psee_lab_obj
)

if(NOT ANDROID)
    # need support for std::filesystem in NDK used GenX320 LowLevel ROI
    list(APPEND lib_camera_obj_list
        metavision_hal_genx320_cx3_system_obj
    )
endif(NOT ANDROID)

if(HAS_V4L2)
    list(APPEND lib_camera_obj_list metavision_hal_v4l2_protocol_obj)
endif(HAS_V4L2)

if(HAL_SAPHIR_SUPPORT)
    list(APPEND lib_camera_obj_list
        metavision_hal_saphir_camera_obj
        metavision_hal_saphir_evk2_system_obj
     )
endif(HAL_SAPHIR_SUPPORT)

if(HAL_SENSORLIB_SUPPORT)
    list(APPEND lib_camera_obj_list
        metavision_hal_sensorlib_tz_obj
        metavision_hal_sensorlib_fx3_obj
     )
endif(HAL_SENSORLIB_SUPPORT)

set(lib_obj_list ${lib_common_obj_list} ${lib_camera_obj_list})

foreach(lib_obj ${lib_obj_list})
    add_library(${lib_obj} OBJECT)
    target_link_libraries(${lib_obj}
        PUBLIC
            metavision_hal
            metavision_psee_hw_layer
    )
    target_include_directories(${lib_obj}
        PUBLIC
            $<BUILD_INTERFACE:${metavision_hal_psee_plugins_include_dir}>
    )

    # The following line is needed because when linking a shared library to an object one,
    # the object library needs to be compiled with -fPIC
    # cf https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
    set_target_properties(${lib_obj}
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            LINKER_LANGUAGE CXX
    )
endforeach()

# Custom target for all plugins (useful to use as a dependency)
add_custom_target(hal_plugins)
add_dependencies(hal_plugins metavision_psee_hw_layer)

    # Otherwise, all the code for live cameras is compiled

    #########################
    # Camera lib objects
    foreach(lib_obj ${lib_camera_obj_list})
        target_link_libraries(${lib_obj}
            PUBLIC
                libusb-1.0
                metavision_psee_hw_layer
        )
    endforeach()

set(hal_psee_plugins_target_file_list)

set(plugin_list
    hal_plugin_prophesee
)

if(HAL_SENSORLIB_SUPPORT)
    list(APPEND plugin_list
        hal_plugin_sensorlib_tz
        hal_plugin_sensorlib_fx3
    )
endif(HAL_SENSORLIB_SUPPORT)

if(POLICY CMP0095)
    # Don't escape RPATH content
    cmake_policy(SET CMP0095 OLD)
endif (POLICY CMP0095)

foreach(plugin ${plugin_list})
    # Declare the plugin as a shared library
    add_library(${plugin} SHARED)
    add_dependencies(hal_plugins ${plugin})

    target_sources(${plugin}
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_psee_plugin_obj>
    )

    target_link_libraries(${plugin}
        PRIVATE
            metavision_psee_hw_layer
            metavision_hal_psee_plugin_obj
        PUBLIC
            metavision_hal
    )

    target_include_directories(${plugin}
        PRIVATE
            ${biasgen_calib_files_dir}
    )

    if(NOT ${CMAKE_VERSION} VERSION_LESS "3.13.3" AND NOT APPLE AND NOT MSVC)
        target_link_options(${plugin} PRIVATE "LINKER:-z,defs")
    endif()

    # Have run-time linker look for metavision_psee_hw_layer library in plugin folder
    get_target_property(plugin_rpath ${plugin} "INSTALL_RPATH")

    if(APPLE)
        set_target_properties(${plugin} PROPERTIES "INSTALL_RPATH" "@loader_path;${plugin_rpath}")
        set_target_properties(${plugin} PROPERTIES "BUILD_RPATH" "@loader_path")
    else ()
        # It seems cmake does an additional variable substitution when replacing BUILD_RPATH with INSTALL_RPATH
        # requiring additional escaping for INSTALL_RPATH
        set_target_properties(${plugin} PROPERTIES "INSTALL_RPATH" "\\\${ORIGIN};${plugin_rpath}")
        set_target_properties(${plugin} PROPERTIES "BUILD_RPATH" "\${ORIGIN}")
    endif()

    list(APPEND hal_psee_plugins_target_file_list $<TARGET_FILE:${plugin}>)

    if(plugin STREQUAL "hal_plugin_sensorlib_tz")
        set(HAL_COPY_PLUGIN_PATH "${HAL_BUILD_PLUGIN_PATH}/sensorlib")
    elseif(plugin STREQUAL "hal_plugin_sensorlib_fx3")
        set(HAL_COPY_PLUGIN_PATH "${HAL_BUILD_PLUGIN_PATH}/sensorlib")
    else()
        set(HAL_COPY_PLUGIN_PATH "${HAL_BUILD_PLUGIN_PATH}")
        install(TARGETS ${plugin}
                LIBRARY DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}" COMPONENT metavision-hal-prophesee-plugins
                RUNTIME DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}" COMPONENT metavision-hal-prophesee-plugins
                #ARCHIVE : we don't want to install .lib files for the plugins, it's useless
        )
    endif()
    # instead of setting the RUNTIME/LIBRARY_OUTPUT_DIRECTORY property on the target, we manually copy
    # the library : this will work for linux and windows and avoid the automatic copy of the DLLs the
    # plugin depends on by MSVC
    add_custom_command(TARGET ${plugin} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_COPY_PLUGIN_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${plugin}>" "${HAL_COPY_PLUGIN_PATH}")

    list(APPEND hal_psee_plugins_target_file_list "${HAL_COPY_PLUGIN_PATH}/$<TARGET_FILE_NAME:${plugin}>")

endforeach()

# Linkage with generic hal plugin object
if(HAL_SENSORLIB_SUPPORT)
    # Linkage with generic hal plugin object
    target_sources(hal_plugin_sensorlib_tz
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_sensorlib_tz_obj>
            $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
            $<TARGET_OBJECTS:metavision_hal_ti_tmp103_obj>
    )

    target_link_libraries(hal_plugin_sensorlib_tz
        PRIVATE
            metavision_hal_sensorlib_tz_obj
            metavision_hal_treuzell_protocol_obj
            metavision_hal_ti_tmp103_obj
    )

    # Linkage with generic hal plugin object
    target_sources(hal_plugin_sensorlib_fx3
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_sensorlib_fx3_obj>
            $<TARGET_OBJECTS:metavision_hal_fx3_protocol_obj>
    )

    target_link_libraries(hal_plugin_sensorlib_fx3
        PRIVATE
            metavision_hal_sensorlib_fx3_obj
            metavision_hal_fx3_protocol_obj
    )
endif(HAL_SENSORLIB_SUPPORT)

foreach(object ${lib_camera_obj_list})
    # Linkage with generic hal plugin object
    target_sources(hal_plugin_prophesee
        PRIVATE
            $<TARGET_OBJECTS:${object}>
    )

    target_link_libraries(hal_plugin_prophesee
        PRIVATE
            ${object}
    )
endforeach()

if(EXISTS "${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK")
    install(FILES ${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK
            DESTINATION share/metavision/licensing)
endif()
