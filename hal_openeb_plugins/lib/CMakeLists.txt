# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_library(hal_raw_common OBJECT)

set_target_properties(hal_raw_common
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.13.3" AND NOT APPLE AND NOT MSVC)
    target_link_options(hal_raw_common PRIVATE "LINKER:-z,defs")
endif()

target_include_directories(hal_raw_common
    PUBLIC
        $<BUILD_INTERFACE:${metavision_hal_openeb_plugins_include_dir}>
)

target_link_libraries(hal_raw_common
    PUBLIC
        metavision_hal
)

# Create one lib per plugin
set(lib_plugin_list
    hal_plugin_gen3_fx3_raw
    hal_plugin_gen4_evk2_raw
    hal_plugin_gen4_fx3_raw
    hal_plugin_gen31_evk2_raw
    hal_plugin_gen31_evk3_raw
    hal_plugin_gen31_fx3_raw
    hal_plugin_gen41_evk2_raw
    hal_plugin_gen41_evk3_raw
)

foreach(lib_plugin ${lib_plugin_list})
    add_library(${lib_plugin} SHARED $<TARGET_OBJECTS:hal_raw_common>)
    target_link_libraries(${lib_plugin} PRIVATE hal_raw_common)
    add_custom_command(TARGET ${lib_plugin} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_BUILD_PLUGIN_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${lib_plugin}>" "${HAL_BUILD_PLUGIN_PATH}")

    if(NOT ${CMAKE_VERSION} VERSION_LESS "3.13.3" AND NOT APPLE AND NOT MSVC)
        target_link_options(${lib_plugin} PRIVATE "LINKER:-z,defs")
    endif()

    install(TARGETS ${lib_plugin}
        LIBRARY DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}" COMPONENT metavision-hal-openeb-plugins EXCLUDE_FROM_ALL
        RUNTIME DESTINATION "${HAL_INSTALL_PLUGIN_RELATIVE_PATH}" COMPONENT metavision-hal-openeb-plugins EXCLUDE_FROM_ALL
        #ARCHIVE : we don't want to install .lib files for the plugins, it's useless
    )
endforeach()
