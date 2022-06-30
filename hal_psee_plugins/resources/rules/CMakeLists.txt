# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if((NOT WIN32) AND (NOT APPLE) AND (NOT ANDROID))

    option(UDEV_RULES_SYSTEM_INSTALL  "Install udev rules on your system files" ON)
    set(rules
        ${CMAKE_CURRENT_SOURCE_DIR}/88-cyusb.rules
        ${CMAKE_CURRENT_SOURCE_DIR}/99-evkv2.rules
    )
    foreach(rule ${rules})
        install(FILES ${rule}
                DESTINATION share/metavision/hal/resources/rules
                COMPONENT metavision-hal-prophesee-plugins
        )
        if(UDEV_RULES_SYSTEM_INSTALL) 
            install(FILES ${rule}
                    DESTINATION /etc/udev/rules.d
                    COMPONENT metavision-hal-prophesee-plugins
            )
    endif()
    endforeach(rule)

endif((NOT WIN32) AND (NOT APPLE) AND (NOT ANDROID))
