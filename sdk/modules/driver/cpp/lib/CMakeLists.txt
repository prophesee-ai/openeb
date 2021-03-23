# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

MetavisionSDK_add_module(driver
    REQUIRED_METAVISION_SDK_MODULES
        PUBLIC
            base
            core
    EXTRA_REQUIRED_PACKAGE MetavisionHAL
)

target_link_libraries(metavision_sdk_driver 
    PUBLIC
        metavision_hal
        metavision_hal_discovery
    PRIVATE
        Boost::filesystem
)