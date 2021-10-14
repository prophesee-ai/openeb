# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

add_executable(raw_file_decoder_with_hal raw_file_decoder_with_hal.cpp)
target_link_libraries(raw_file_decoder_with_hal PRIVATE metavision_hal_discovery)

add_test_app(metavision_evt2_raw_file_decoder)
add_test_app(metavision_evt3_raw_file_decoder)
add_test_app(metavision_evt2_raw_file_encoder)