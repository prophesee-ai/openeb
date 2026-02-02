# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

MetavisionSDK_add_module(core
    REQUIRED_METAVISION_SDK_MODULES
        PUBLIC
            base
    EXTRA_REQUIRED_PACKAGE "OpenCV COMPONENTS core imgproc highgui videoio"
    EXTRA_REQUIRED_PACKAGE "Boost COMPONENTS timer"
    EXTRA_REQUIRED_PACKAGE "Threads"
)

# VideoWriter needs code extracted from 3rdparty OpenCV v4.5 which is in src folder
target_include_directories(metavision_sdk_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(metavision_sdk_core
    PUBLIC
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_videoio
        Boost::boost # Target for header-only dependencies
        Boost::timer
        Threads::Threads
)