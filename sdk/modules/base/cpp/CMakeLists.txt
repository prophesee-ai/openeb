# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Add library
add_subdirectory(lib)

# Add sources to the library
add_subdirectory(src)

# Samples
if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif (BUILD_SAMPLES)

# Documentations
if (GENERATE_DOC)
    add_subdirectory(doc)
endif (GENERATE_DOC)

# Tests
if (BUILD_TESTING)
    add_subdirectory(tests)
endif (BUILD_TESTING)

add_cpack_component(PUBLIC metavision-sdk-base-lib metavision-sdk-base-dev metavision-sdk-base-bin metavision-sdk-base-samples)
