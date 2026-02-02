# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

###################################################
# Function to easily wrap Android gradle commands
function(add_android_app app)
    set(options)
    set(multiValueArgs DEPENDS EXCLUDE_GRADLE_TASKS)
    cmake_parse_arguments(PARSED_ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSED_ARGS_DEPENDS)
        set(_depends ${PARSED_ARGS_DEPENDS})
    endif(PARSED_ARGS_DEPENDS)

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/local.properties.in"
        "${CMAKE_CURRENT_BINARY_DIR}/local.properties"
        @ONLY
    )
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/local.properties"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/local.properties"
    )

    set(ANDROID_GRADLE_CACHE_EXTRACT_DIR ${GENERATE_FILES_DIRECTORY}/android)
    set(ANDROID_GRADLE_CACHE_DIR ${ANDROID_GRADLE_CACHE_EXTRACT_DIR}/.gradle2)
    set(gradle_options "--project-cache-dir;${CMAKE_CURRENT_BINARY_DIR}/.gradle;-g;${ANDROID_GRADLE_CACHE_DIR};")
    list(APPEND gradle_options "--console=plain" "--info")

    if(PARSED_ARGS_EXCLUDE_GRADLE_TASKS)
        list(APPEND gradle_options "-x" ${PARSED_ARGS_EXCLUDE_GRADLE_TASKS})
    endif(PARSED_ARGS_EXCLUDE_GRADLE_TASKS)

    if (GRADLE_OFFLINE_MODE)
        # Unpack gradle cache for faster and consistent builds
        set(ANDROID_GRADLE_CACHE_ARCHIVE utils/android/gradle-cache.tar.gz)
        if (NOT EXISTS ${ANDROID_GRADLE_CACHE_DIR})
            lfs_download(COMPILATION IMMEDIATE ${ANDROID_GRADLE_CACHE_ARCHIVE})
            message(STATUS "Unpacking ${ANDROID_GRADLE_CACHE_ARCHIVE} in ${ANDROID_GRADLE_CACHE_EXTRACT_DIR}")
            file(MAKE_DIRECTORY ${ANDROID_GRADLE_CACHE_EXTRACT_DIR})
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar -xf ${PROJECT_SOURCE_DIR}/${ANDROID_GRADLE_CACHE_ARCHIVE}
                WORKING_DIRECTORY ${ANDROID_GRADLE_CACHE_EXTRACT_DIR}
            )
        endif (NOT EXISTS ${ANDROID_GRADLE_CACHE_DIR})
        list(APPEND gradle_options "--offline")
    endif (GRADLE_OFFLINE_MODE)

    add_custom_target(${app} ALL
        COMMAND ./gradlew ${gradle_options} build
        COMMAND ./gradlew ${gradle_options} assembleAndroidTest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )

    set (target deploy_${app})
    add_custom_target(${target}
        COMMAND ./gradlew ${gradle_options} uninstall${CMAKE_BUILD_TYPE}
        COMMAND ./gradlew ${gradle_options} install${CMAKE_BUILD_TYPE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(${target} ${app})

    set (target clean_${app})
    add_custom_target(${target}
            COMMAND ./gradlew ${gradle_options} clean
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set (target test_${app})
    add_custom_target(${target}
            COMMAND ./gradlew ${gradle_options} "-i" connectedAndroidTest
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(PARSED_ARGS_DEPENDS)
        add_dependencies(${app} ${_depends})
        add_dependencies(${target} ${_depends})
    endif(PARSED_ARGS_DEPENDS)

endfunction(add_android_app app)
