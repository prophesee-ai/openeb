# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.6)

project(metavision)

set(CPACK_PACKAGE_VENDOR "Prophesee")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "support@prophesee.ai")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE https://support.prophesee.ai)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE all)
set(CPACK_DEBIAN_PACKAGE_SECTION base)

set (PACKAGE_LICENSE "License : Copyright (c) Prophesee S.A. - All Rights Reserved")


# This script is to be able to create a debian package from a folder or a tar archive
#
# Usage :
#
#    1) create a build folder :
#
#       mkdir _build_deb
#
#
#    2) inside the build folder, run cmake
#
#       cmake ${CMAKE_OPTIONS} ${PATH_FOLDER_WHERE_THIS_FILE_IS}
#
#
#    3) inside the build folder, run the target to create the package :
#
#       make debian_package
#
#
#
#     - PATH_FOLDER_WHERE_THIS_FILE_IS = path to the directory of this file (the folder create_deb_package_from_folder)
#
#     - CMAKE_OPTIONS : they specify the options for creating the debian package.
#
#       Required options :
#
#           - one between SOURCE_DIRECTORY and SOURCE_ARCHIVE :
#                 SOURCE_DIRECTORY = path to the folder to use to generate the debian package
#                 SOURCE_ARCHIVE   = path to the tar archive from which to generate the debian package.
#
#
#       Optional options :
#
#           - OUTPUT_PACKAGE_NAME, OUTPUT_PACKAGE_VERSION = respectively the (installation) name and version of the
#                 output debian package. If not provided, they will be automatically determined from the name of the
#                 input folder/archive, provided that it's of the form
#                               ${OUTPUT_PACKAGE_NAME}-${OUTPUT_PACKAGE_VERSION}                  if input is folder
#                               ${OUTPUT_PACKAGE_NAME}-${OUTPUT_PACKAGE_VERSION}.{some_extention} if input is archive
#                 Otherwise, these options are mandatory
#
#           - OUTPUT_FILE_NAME = name of the package file to generate, not including the extension. Default value is
#                 ${OUTPUT_PACKAGE_NAME}-${OUTPUT_PACKAGE_VERSION}
#
#           - OUTPUT_PACKAGE_DESCRIPTION = Package description. Default value is : "Metavision's package ${OUTPUT_PACKAGE_NAME}"
#
#
# Example : let's say you have a file /home/bamboo/metavision-sdk-analytics-dataset-2.0.0.tgz
#
#   mkdir _build_dataset
#   cd _build_dataset
#   cmake -DSOURCE_ARCHIVE=/home/bamboo/metavision-sdk-analytics-dataset-2.0.0.tgz
#       -DOUTPUT_PACKAGE_DESCRIPTION="Dataset for Metavision SDK Analytics code samples" ../utils/cmake/create_deb_package_from_folder/
#   make debian_package
#
#  will generate debian package : metavision-sdk-analytics-dataset-2.0.0.deb
#


# Check input args :
if((NOT DEFINED SOURCE_DIRECTORY) AND (NOT DEFINED SOURCE_ARCHIVE))
    message(FATAL_ERROR "Error : you need to provide either SOURCE_DIRECTORY or SOURCE_ARCHIVE option")
endif((NOT DEFINED SOURCE_DIRECTORY) AND (NOT DEFINED SOURCE_ARCHIVE))
if((DEFINED SOURCE_DIRECTORY) AND (DEFINED SOURCE_ARCHIVE))
    message(FATAL_ERROR "Error : options SOURCE_DIRECTORY are SOURCE_ARCHIVE are mutually exclusive")
endif((DEFINED SOURCE_DIRECTORY) AND (DEFINED SOURCE_ARCHIVE))
if(DEFINED SOURCE_DIRECTORY)
    if(NOT IS_DIRECTORY ${SOURCE_DIRECTORY})
        message(FATAL_ERROR "Error : input folder ${SOURCE_DIRECTORY} does not exist")
    endif(NOT IS_DIRECTORY ${SOURCE_DIRECTORY})
else()
endif(DEFINED SOURCE_DIRECTORY)

# Extract information from source archive automatically if not provided :
if((DEFINED OUTPUT_PACKAGE_NAME) AND (DEFINED OUTPUT_PACKAGE_VERSION))
    set(CPACK_PACKAGE_NAME "${OUTPUT_PACKAGE_NAME}")
    set(CPACK_PACKAGE_VERSION "${OUTPUT_PACKAGE_VERSION}")
else()
    if(DEFINED SOURCE_DIRECTORY)
        get_filename_component(source_dir_name ${SOURCE_DIRECTORY} NAME)
        if("${source_dir_name}" MATCHES "^(.+)-([0-9]+\\.[0-9]+\\.[0-9]+)$")
            set(CPACK_PACKAGE_NAME "${CMAKE_MATCH_1}")
            set(CPACK_PACKAGE_VERSION "${CMAKE_MATCH_2}")
        else()
            message(FATAL_ERROR "Error : you need to provide DEB_NAME and DEB_VERSION, because they cannot be inferred by input ${SOURCE_DIRECTORY}")
        endif("${source_dir_name}" MATCHES "(.+)-([0-9]+\\.[0-9]+\\.[0-9]+)\\.+")
    else()
        get_filename_component(source_archive_name ${SOURCE_ARCHIVE} NAME)
        if("${source_archive_name}" MATCHES "^(.+)-([0-9]+\\.[0-9]+\\.[0-9]+)\\.+")
            set(CPACK_PACKAGE_NAME "${CMAKE_MATCH_1}")
            set(CPACK_PACKAGE_VERSION "${CMAKE_MATCH_2}")
        else()
            message(FATAL_ERROR "Error : you need to provide DEB_NAME and DEB_VERSION, because they cannot be inferred by input ${SOURCE_ARCHIVE}")
        endif("${source_archive_name}" MATCHES "^(.+)-([0-9]+\\.[0-9]+\\.[0-9]+)\\.+")
    endif(DEFINED SOURCE_DIRECTORY)
endif((DEFINED OUTPUT_PACKAGE_NAME) AND (DEFINED OUTPUT_PACKAGE_VERSION))

if(DEFINED OUTPUT_FILE_NAME)
    set(CPACK_PACKAGE_FILE_NAME "${OUTPUT_FILE_NAME}")
else()
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
endif(DEFINED OUTPUT_FILE_NAME)

if (DEFINED OUTPUT_PACKAGE_DESCRIPTION)
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${OUTPUT_PACKAGE_DESCRIPTION}\n${PACKAGE_LICENSE}")
else()
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Metavision's package ${OUTPUT_PACKAGE_NAME}\n${PACKAGE_LICENSE}")
endif(DEFINED OUTPUT_PACKAGE_DESCRIPTION)

if (DEFINED SOURCE_DIRECTORY)

    install(DIRECTORY ${SOURCE_DIRECTORY}/
            DESTINATION .
    )
    add_custom_target(debian_package
                      COMMAND ${CMAKE_CPACK_COMMAND} -G DEB
                      COMMENT "Generating debian package from folder ${SOURCE_DIRECTORY}"
    )
else()

    set(extract_dirname "${CMAKE_BINARY_DIR}/tmpdir_package_contents")

    # Create tmp dir
    add_custom_target(create_folder_to_extract_archive
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${extract_dirname}" # clean
        COMMAND ${CMAKE_COMMAND} -E make_directory "${extract_dirname}"
        COMMENT "Creating temporary directory ${extract_dirname}"
    )

    # Extract archive
    get_filename_component(full_path_to_source "${SOURCE_ARCHIVE}" ABSOLUTE)
    add_custom_target(extract_archive
        COMMAND ${CMAKE_COMMAND} -E tar xvf ${full_path_to_source}
        COMMENT "Extracting file ${full_path_to_source} into ${extract_dirname}"
        WORKING_DIRECTORY ${extract_dirname}
    )
    add_dependencies(extract_archive create_folder_to_extract_archive)


    # This folder will actually be created at build time
    install(DIRECTORY ${extract_dirname}/
            DESTINATION .
    )

    add_custom_target(debian_package
                      COMMAND ${CMAKE_CPACK_COMMAND} -G DEB
                      COMMAND ${CMAKE_COMMAND} -E make_directory "${extract_dirname}"
                      COMMENT "Generating debian package from archive ${full_path_to_source}"
    )
    add_dependencies(debian_package extract_archive)

endif(DEFINED SOURCE_DIRECTORY)

include(CPack)

