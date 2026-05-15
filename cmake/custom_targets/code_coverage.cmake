# Copyright (c) 2012 - 2015, Lars Bilke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#
# 2012-01-31, Lars Bilke
# - Enable Code Coverage
#
# 2013-09-17, Joakim Söderberg
# - Added support for Clang.
# - Some additional usage instructions.
#

# Check prereqs
find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)

if(NOT GCOV_PATH)
    message(FATAL_ERROR "gcov not found! Aborting...")
endif() # NOT GCOV_PATH

if(NOT CMAKE_COMPILER_IS_GNUCXX)
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        message(FATAL_ERROR "Compiler is not GNU gcc nor clang ! Aborting...")
    endif()
endif()

set(CodeCoverage_FOUND ON)

set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 -fprofile-arcs -ftest-coverage -fno-inline" CACHE STRING "Flags used by the C++ compiler during coverage builds." FORCE)
set(CMAKE_C_FLAGS_COVERAGE "-g -O0 -fprofile-arcs -ftest-coverage -fno-inline" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE} -fno-inline-small-functions -fno-default-inline" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE)
    set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_COVERAGE} -fno-inline-small-functions -fno-default-inline" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE)
endif()
mark_as_advanced(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
)
        
add_custom_target(coverage
    # Cleanup lcov
    COMMAND ${LCOV_PATH} -d ${PROJECT_SOURCE_DIR} -z
    # Capturing lcov counters and generating report (before running tests)
    COMMAND ${LCOV_PATH} --no-external -d ${PROJECT_SOURCE_DIR} -c -i -o coverage-before.info
    # Run tests
    COMMAND ${CMAKE_CTEST_COMMAND}
    # Capturing lcov counters and generating report (after running tests)
    COMMAND ${LCOV_PATH} --no-external -d ${PROJECT_SOURCE_DIR} -c -o coverage-after.info
    # Combine both info files
    COMMAND ${LCOV_PATH} -a coverage-before.info -a coverage-after.info -o coverage-merged.info
    # Remove specific files (tests, etc.)
    COMMAND ${LCOV_PATH} -r coverage-merged.info "*gtest*.cpp" -o coverage.info
    # Generate html report
    COMMAND ${GENHTML_PATH} -o coverage coverage.info
    # Remove generated files
    COMMAND ${CMAKE_COMMAND} -E remove coverage-before.info coverage-after.info coverage-merged.info coverage.info
    # Show info where to find the report
    COMMAND ${CMAKE_COMMAND} -E echo "Open coverage/index.html in your browser to view the coverage report."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Resetting code coverage counters to zero, processing code coverage counters and generating report."
)

message(STATUS "Code coverage enabled, run the coverage target to access it")
