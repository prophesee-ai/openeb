# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

include(CMakeParseArguments)

# Function to embed contents of a file as byte array in C/C++ format. This function will read a file
# and convert it into a text string of hex values (syntax compatible with C++ and other similar
# languages). The output is placed in variables (as return data of this function).
#
# Parameters
#   SOURCE_FILE     - The path of source file whose contents will be embedded in the header file.
#   VARIABLE_NAME   - The base name of the variables to output. The byte array data string will be placed
#                     in "${VARIABLE_NAME}_DATA". The length (size) of the binary data will be placed in
#                     "${VARIABLE_NAME}_SIZE".
# Usage:
#   bin2h(SOURCE_FILE "data.bin" VARIABLE_NAME MY_DATA)
#
#   This call will generate: MY_DATA_DATA and MY_DATA_SIZE variables.
function(BIN2H)
    set(oneValueArgs SOURCE_FILE VARIABLE_NAME)
    cmake_parse_arguments(BIN2H "${options}" "${oneValueArgs}" "" ${ARGN})

    # Reads source file contents as hex string.
    file(READ "${BIN2H_SOURCE_FILE}" hexString HEX)
    string(LENGTH ${hexString} hexStringLength)
    math(EXPR arraySize "${hexStringLength} / 2")

    # Split the content byte per byte (pairs of 2 chars).
    string(REGEX MATCHALL ".." hexString "${hexString}")
    # ... then split the array every 16 bytes and output the chunk on a single line ...
    string(REGEX REPLACE "(..;..;..;..;..;..;..;..;..;..;..;..;..;..;..;..);" "\\1,\n0x" hexString "${hexString}")
    # ... then prefix bytes with 0x
    string(REGEX REPLACE ";" ", 0x" hexString "${hexString}")

    # Return.
    set(${BIN2H_VARIABLE_NAME}_SIZE ${arraySize} PARENT_SCOPE)
    set(${BIN2H_VARIABLE_NAME}_DATA "0x${hexString}" PARENT_SCOPE)
endfunction()
