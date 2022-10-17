# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_cx3_updater
    metavision_cx3_updater.cpp
    ${PROJECT_SOURCE_DIR}/hal_psee_plugins/src/boards/utils/utils_fx3_ram_flash.cpp
    ${PROJECT_SOURCE_DIR}/hal_psee_plugins/src/devices/others/i2c_eeprom.cpp
)
target_link_libraries(metavision_cx3_updater PRIVATE MetavisionSDK::base Boost::program_options libusb-1.0)
target_include_directories(metavision_cx3_updater 
    PRIVATE
        $<TARGET_PROPERTY:metavision_hal_psee_plugin_obj,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS metavision_cx3_updater
        RUNTIME DESTINATION bin
        COMPONENT metavision-hal-prophesee-plugins
)

