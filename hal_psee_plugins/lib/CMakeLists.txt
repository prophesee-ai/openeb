# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Compile code that is used multiple times in object libraries :

set(lib_common_obj_list
        metavision_hal_psee_plugin_obj
)

set(lib_camera_obj_list "")
if (NOT ANDROID)
    set(lib_camera_obj_list
            metavision_hal_gen3_camera_obj
            metavision_hal_gen3_pattern_checkers_obj
            metavision_hal_gen31_camera_obj
            metavision_hal_gen41_camera_obj
            metavision_hal_imx636_camera_obj
            metavision_hal_fx3_protocol_obj
            metavision_hal_treuzell_protocol_obj
            metavision_hal_ccam3_obj
            metavision_hal_psee_video_obj
            metavision_hal_psee_video_build_obj
            metavision_hal_ti_tmp103_obj
            metavision_hal_gen31_fx3_system_obj
            metavision_hal_gen31_evk2_system_obj
            metavision_hal_gen31_evk3_system_obj
            metavision_hal_gen41_evk2_system_obj
            metavision_hal_gen41_evk3_system_obj
            metavision_hal_imx636_evk2_system_obj
            metavision_hal_imx636_evk3_system_obj
            metavision_hal_libusb_utils_obj
    )
    if(HAL_GEN4_SUPPORT)
        list(APPEND lib_camera_obj_list
            metavision_hal_gen4_camera_obj
            metavision_hal_gen4_fx3_system_obj
            metavision_hal_gen4_evk2_system_obj
         )
    endif(HAL_GEN4_SUPPORT)
endif (NOT ANDROID)

set(lib_obj_list ${lib_common_obj_list} ${lib_camera_obj_list})

foreach(lib_obj ${lib_obj_list})
    add_library(${lib_obj} OBJECT)
    target_link_libraries(${lib_obj}
        PUBLIC
            metavision_hal
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
    )
endforeach()

# Custom target for all plugins (useful to use as a dependency)
add_custom_target(hal_plugins)

# Android does not support live cameras -> we need to compile only the common objects for RAW file reading.
if (NOT ANDROID)
    # Otherwise, all the code for live cameras is compiled

    #########################
    # Camera lib objects
    foreach(lib_obj ${lib_camera_obj_list})
        target_link_libraries(${lib_obj}
            PUBLIC
                libusb-1.0
        )

        target_include_directories(${lib_obj}
            PRIVATE
                ${biasgen_calib_files_dir}
        )
    endforeach()
endif (NOT ANDROID)

set(hal_psee_plugins_target_file_list)

set(plugin_list
    hal_plugin_gen3_fx3
    hal_plugin_gen31_fx3
    hal_plugin_gen31_evk2
    hal_plugin_gen31_evk3
    hal_plugin_gen41_evk2
    hal_plugin_gen41_evk3
    hal_plugin_imx636_evk2
    hal_plugin_imx636_evk3
    hal_plugin_imx636_evk4
    hal_plugin_golden_fallbacks
    hal_plugin_prophesee
)

if(HAL_GEN4_SUPPORT)
  list(APPEND plugin_list
      hal_plugin_gen4_fx3
      hal_plugin_gen4_evk2
  )
endif(HAL_GEN4_SUPPORT)

foreach(plugin ${plugin_list})
    # Declare the plugin as a shared library
    add_library(${plugin} SHARED)
    add_dependencies(hal_plugins ${plugin})

    target_sources(${plugin}
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_psee_plugin_obj>
            $<TARGET_OBJECTS:metavision_hal_info_obj>
    )

    target_link_libraries(${plugin}
        PRIVATE
            metavision_hal_psee_plugin_obj
            metavision_hal_info_obj
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

    list(APPEND hal_psee_plugins_target_file_list $<TARGET_FILE:${plugin}>)

    if(plugin STREQUAL "hal_plugin_prophesee")
        set(HAL_COPY_PLUGIN_PATH "${HAL_BUILD_PLUGIN_PATH}/universal")
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

if (NOT ANDROID)
# Linkage with generic hal plugin object
target_sources(hal_plugin_gen3_fx3
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen3_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_gen3_pattern_checkers_obj>
        $<TARGET_OBJECTS:metavision_hal_fx3_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ccam3_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen3_fx3
    PRIVATE
        metavision_hal_gen3_camera_obj
        metavision_hal_gen3_pattern_checkers_obj
        metavision_hal_fx3_protocol_obj
        metavision_hal_ccam3_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_gen31_fx3
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen31_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_gen3_pattern_checkers_obj>
        $<TARGET_OBJECTS:metavision_hal_fx3_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ccam3_obj>
        $<TARGET_OBJECTS:metavision_hal_gen31_fx3_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen31_fx3
    PRIVATE
        metavision_hal_gen31_camera_obj
        metavision_hal_gen3_pattern_checkers_obj
        metavision_hal_fx3_protocol_obj
        metavision_hal_ccam3_obj
        metavision_hal_gen31_fx3_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_gen31_evk2
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen31_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_gen3_pattern_checkers_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_psee_video_obj>
        $<TARGET_OBJECTS:metavision_hal_gen31_evk2_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen31_evk2
    PRIVATE
        metavision_hal_gen31_camera_obj
        metavision_hal_gen3_pattern_checkers_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_psee_video_obj
        metavision_hal_gen31_evk2_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_gen31_evk3
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen31_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_gen3_pattern_checkers_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ti_tmp103_obj>
        $<TARGET_OBJECTS:metavision_hal_gen31_evk3_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen31_evk3
    PRIVATE
        metavision_hal_gen31_camera_obj
        metavision_hal_gen3_pattern_checkers_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_ti_tmp103_obj
        metavision_hal_gen31_evk3_system_obj
        metavision_hal_libusb_utils_obj
)

if(HAL_GEN4_SUPPORT)
    # Linkage with generic hal plugin object
    target_sources(hal_plugin_gen4_fx3
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_gen4_camera_obj>
            $<TARGET_OBJECTS:metavision_hal_fx3_protocol_obj>
            $<TARGET_OBJECTS:metavision_hal_ccam3_obj>
            $<TARGET_OBJECTS:metavision_hal_gen4_fx3_system_obj>
            $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
    )

    target_link_libraries(hal_plugin_gen4_fx3
        PRIVATE
            metavision_hal_gen4_camera_obj
            metavision_hal_fx3_protocol_obj
            metavision_hal_ccam3_obj
            metavision_hal_gen4_fx3_system_obj
            metavision_hal_libusb_utils_obj
    )

    # Linkage with generic hal plugin object
    target_sources(hal_plugin_gen4_evk2
        PRIVATE
            $<TARGET_OBJECTS:metavision_hal_gen4_camera_obj>
            $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
            $<TARGET_OBJECTS:metavision_hal_psee_video_obj>
            $<TARGET_OBJECTS:metavision_hal_gen4_evk2_system_obj>
            $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
    )

    target_link_libraries(hal_plugin_gen4_evk2
        PRIVATE
            metavision_hal_gen4_camera_obj
            metavision_hal_treuzell_protocol_obj
            metavision_hal_psee_video_obj
            metavision_hal_gen4_evk2_system_obj
            metavision_hal_libusb_utils_obj
    )
endif(HAL_GEN4_SUPPORT)

# Linkage with generic hal plugin object
target_sources(hal_plugin_gen41_evk2
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen41_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_psee_video_obj>
        $<TARGET_OBJECTS:metavision_hal_gen41_evk2_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen41_evk2
    PRIVATE
        metavision_hal_gen41_camera_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_psee_video_obj
        metavision_hal_gen41_evk2_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_gen41_evk3
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen41_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ti_tmp103_obj>
        $<TARGET_OBJECTS:metavision_hal_gen41_evk3_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_gen41_evk3
    PRIVATE
        metavision_hal_gen41_camera_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_ti_tmp103_obj
        metavision_hal_gen41_evk3_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_imx636_evk2
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen41_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_psee_video_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_evk2_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_imx636_evk2
    PRIVATE
        metavision_hal_gen41_camera_obj
        metavision_hal_imx636_camera_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_psee_video_obj
        metavision_hal_imx636_evk2_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_imx636_evk3
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen41_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ti_tmp103_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_evk3_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_imx636_evk3
    PRIVATE
        metavision_hal_gen41_camera_obj
        metavision_hal_imx636_camera_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_ti_tmp103_obj
        metavision_hal_imx636_evk3_system_obj
        metavision_hal_libusb_utils_obj
)

# Linkage with generic hal plugin object
target_sources(hal_plugin_imx636_evk4
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_gen41_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_camera_obj>
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_ti_tmp103_obj>
        $<TARGET_OBJECTS:metavision_hal_imx636_evk3_system_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_imx636_evk4
    PRIVATE
        metavision_hal_gen41_camera_obj
        metavision_hal_imx636_camera_obj
        metavision_hal_treuzell_protocol_obj
        metavision_hal_ti_tmp103_obj
        metavision_hal_imx636_evk3_system_obj
        metavision_hal_libusb_utils_obj
)

target_sources(hal_plugin_golden_fallbacks
    PRIVATE
        $<TARGET_OBJECTS:metavision_hal_treuzell_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_fx3_protocol_obj>
        $<TARGET_OBJECTS:metavision_hal_libusb_utils_obj>
)

target_link_libraries(hal_plugin_golden_fallbacks
    PRIVATE
        metavision_hal_treuzell_protocol_obj
        metavision_hal_fx3_protocol_obj
        metavision_hal_libusb_utils_obj
)

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

endif(NOT ANDROID)

if(EXISTS "${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_INTELLIGENCE")
    install(FILES ${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_INTELLIGENCE
            DESTINATION share/metavision/licensing)
endif()
