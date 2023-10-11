# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(metavision_hal_seek metavision_hal_seek.cpp)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(metavision_hal_seek PRIVATE metavision_hal_discovery Boost::program_options ${OpenCV_LIBS} Threads::Threads)

install(FILES metavision_hal_seek.cpp
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_seek
        COMPONENT metavision-hal-samples
)

install(FILES CMakeLists.txt.install
        RENAME CMakeLists.txt
        DESTINATION share/metavision/hal/cpp_samples/metavision_hal_seek
        COMPONENT metavision-hal-samples
)
