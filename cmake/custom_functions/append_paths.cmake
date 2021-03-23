# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

function(append_paths output_var path1 path2)

    if(WIN32)
        # Need to escape because cmake interprets the semi-colon as list separator
        set(PATHS_SEPARATOR_CHAR "\;")
        string(REPLACE ";" "${PATHS_SEPARATOR_CHAR}" path1 "${path1}")
        string(REPLACE ";" "${PATHS_SEPARATOR_CHAR}" path2 "${path2}")
    else() # i.e NOT WIN32
        set(PATHS_SEPARATOR_CHAR ":")
    endif(WIN32)

    # Now append paths
    set(result "${path1}${PATHS_SEPARATOR_CHAR}${path2}")
    foreach(p ${ARGN})
        if(WIN32)
            string(REPLACE ";" "${PATHS_SEPARATOR_CHAR}" p "${p}")
        endif(WIN32)
        set(result "${result}${PATHS_SEPARATOR_CHAR}${p}")
    endforeach(p)
    set(${output_var} "${result}" PARENT_SCOPE)

endfunction(append_paths)