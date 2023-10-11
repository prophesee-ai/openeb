# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Compile the samples to check they compile
set(standalone_sample
    metavision_evt2_raw_file_decoder
    metavision_evt3_raw_file_decoder
    metavision_evt2_raw_file_encoder
)

foreach(sample ${standalone_sample})
    add_executable(${sample} ${sample}/${sample}.cpp)

    # Install sample in package metavision-decoders-samples
    install(DIRECTORY ${sample}
            DESTINATION share/metavision/standalone_samples
            COMPONENT metavision-decoders-samples
    )
endforeach(sample)

# cpack
add_cpack_component(PUBLIC metavision-decoders-samples)

# Test applications
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)