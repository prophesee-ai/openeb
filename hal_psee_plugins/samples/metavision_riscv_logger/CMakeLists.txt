# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

set(sample metavision_riscv_logger)
set(common_libraries metavision_hal_discovery MetavisionSDK::base Boost::program_options)

add_executable(metavision_riscv_logger metavision_riscv_logger.cpp)
target_link_libraries(metavision_riscv_logger PRIVATE ${common_libraries})

install(TARGETS ${sample}
    RUNTIME DESTINATION bin
    COMPONENT metavision-sdk-stream-bin
)

install(FILES
        ${sample}.cpp 
        hello_world.hex
    DESTINATION share/metavision/hal_psee_plugins/cpp_samples/${sample}
    COMPONENT metavision-hal-prophesee-plugins
)

install(FILES CMakeLists.txt.install
    RENAME CMakeLists.txt
    DESTINATION share/metavision/hal_psee_plugins/cpp_samples/${sample}
    COMPONENT metavision-hal-prophesee-plugins
)