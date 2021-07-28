# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.5)

project(metavision VERSION 2.2.2)

# Set output directory for build targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/ CACHE PATH "Output directory of libraries.")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/ CACHE PATH "Output directory of all executables.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/ CACHE PATH "Output directory of all executables.")
set(PYTHON3_OUTPUT_DIR ${PROJECT_BINARY_DIR}/py3/ CACHE PATH "Output directory of python 3 libraries.")

# Set directory for Metavision and HAL plugins
if (WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        string(REGEX REPLACE "${PROJECT_NAME}$" "Prophesee" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
        set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "" FORCE)
    endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(WIN32)
set(HEADER_INSTALL_DEST "include")
if (ANDROID)
    set(LIBRARY_INSTALL_DEST "libs/${ANDROID_ABI}")
    set(ARCHIVE_INSTALL_DEST "libs/${ANDROID_ABI}")
    set(RUNTIME_INSTALL_DEST "bins/${ANDROID_ABI}")
    set(HAL_INSTALL_PLUGIN_RELATIVE_PATH "libs/${ANDROID_ABI}/metavision/hal/plugins")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/metavision-${PROJECT_VERSION}")
else (ANDROID)
    set(LIBRARY_INSTALL_DEST "lib")
    set(ARCHIVE_INSTALL_DEST "lib")
    set(RUNTIME_INSTALL_DEST "bin")
    set(HAL_INSTALL_PLUGIN_RELATIVE_PATH "lib/metavision/hal/plugins")
endif (ANDROID)
set(HAL_BUILD_PLUGIN_PATH "${PROJECT_BINARY_DIR}/${HAL_INSTALL_PLUGIN_RELATIVE_PATH}")

# Set output directory for generated files
set(GENERATE_FILES_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATE_FILES_DIRECTORY})

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Update CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules) # For find_packages()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/custom_functions) # For defining custom cmake functions
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/custom_targets) # For defining custom cmake targets
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/cpack) # For cpack

include(CMakeDependentOption)
include(overridden_cmake_functions)
include(lfs_download)
if (ANDROID)
    # Setup 3rd party prebuilt libraries dir
    set(ANDROID_PREBUILT_3RDPARTY_ARCHIVE utils/android/prebuilt-3rdparty.tar.gz)
    set(ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR ${GENERATE_FILES_DIRECTORY}/android/3rdparty)
    if (NOT EXISTS ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})
        lfs_download(${ANDROID_PREBUILT_3RDPARTY_ARCHIVE})
        message(STATUS "Unpacking ${ANDROID_PREBUILT_3RDPARTY_ARCHIVE} in ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR}")
        file(MAKE_DIRECTORY ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar -xf ${PROJECT_SOURCE_DIR}/${ANDROID_PREBUILT_3RDPARTY_ARCHIVE}
            WORKING_DIRECTORY ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR}
        )
    endif (NOT EXISTS ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})

    # Include the function required to build an Android APK
    include(add_android_app)

    # Use Android env.cmake to find required dependencies
    set(ANDROID_PREBUILT_3RDPARTY_DIR ${GENERATE_FILES_DIRECTORY}/android/3rdparty/prebuilt)
    include(${PROJECT_SOURCE_DIR}/cmake/android/env.cmake)
endif (ANDROID)

###################################################

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################################### Platform specific

if (MSVC)
    set(CMAKE_DEBUG_POSTFIX "_d")

    if(${CMAKE_VERSION} VERSION_LESS "3.12")
        message(ERROR "Building with Microsoft Visual Studio requires CMake 3.12 or later!")
    endif()

    # Add definition for missing macros
    add_definitions("-D__PRETTY_FUNCTION__=__FUNCSIG__")

    # For including cmath
    # Cf https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants?redirectedfrom=MSDN&view=vs-2019
    add_definitions("-D_USE_MATH_DEFINES")

    # Disable Boost auto dynamic linking.
    add_definitions("-DBOOST_ALL_NO_LIB")

    # This is fine.
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)

    # These warnings indicate loss of data by implicitly casting types (like 64-bit ints) to smaller types. We
    # know about this and expect that the affected code is nevertheless correct. So, disabling these will give
    # more readable compiler output under Visual Studio (prevents warnings flooding the output).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd\\\"4244\\\" /wd\\\"4267\\\"")

    # Export every symbol when building DLLs. This is similar to the behavior of GCC on Linux.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # Since vcpkg does not automatically include top level include path
    # and to be able to include some "system files", like dirent.h
    find_path(VCPKG_INCLUDE_DIR dirent.h)
    include_directories(${VCPKG_INCLUDE_DIR})
endif()

if (WIN32)
    # We need to use '\\' instead of '/' for subkeys, as those are not file paths
    set(METAVISION_SUBKEY "Software\\Prophesee")
    set(METAVISION_SUBKEY_INSTALL_PATH "INSTALL_PATH")

    # replace '\\' by '\\\\' to finally get '\\' in the generated header files later in the code
    string(REPLACE "\\" "\\\\" METAVISION_SUBKEY ${METAVISION_SUBKEY})
endif()

################################################### CMake options

if (NOT ANDROID)
    option(COMPILE_PYTHON3_BINDINGS "Compile python 3 bindings" ON)
    cmake_dependent_option(CODE_COVERAGE "Enable code coverage" ON "CMAKE_BUILD_TYPE_LOWER STREQUAL debug" OFF)
    string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
    cmake_dependent_option(GENERATE_DOC "Generate Doxygen documentation" OFF "COMPILE_PYTHON3_BINDINGS" OFF)
    cmake_dependent_option(GENERATE_DOC_PYTHON_BINDINGS "Generate python bindings documentation from C++" ON "GENERATE_DOC" OFF)
else (NOT ANDROID)
    option(GRADLE_OFFLINE_MODE "Gradle will not try to download dependencies (assumes the cache is already filled)" OFF)
endif (NOT ANDROID)
set(DATASET_DIR "" CACHE PATH "Folder with dataset for testing")

################################################### Detect which SDK modules are available

set(METAVISION_SDK_MODULES_OPEN base core driver)
set(METAVISION_SDK_MODULES_ESSENTIALS analytics calibration cv ml)
if (NOT ANDROID)
    list(APPEND METAVISION_SDK_MODULES_OPEN ui)
    list(APPEND METAVISION_SDK_MODULES_ESSENTIALS 3dview cv3d)
endif (NOT ANDROID)

foreach(metavision_offer OPEN ESSENTIALS)
    set(METAVISION_SDK_${metavision_offer}_MODULES_AVAILABLE)
    foreach(module_name IN LISTS METAVISION_SDK_MODULES_${metavision_offer})
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/sdk/modules/${module_name}/CMakeLists.txt")
            list(APPEND METAVISION_SDK_${metavision_offer}_MODULES_AVAILABLE "${module_name}")
        endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/sdk/modules/${module_name}/CMakeLists.txt")
    endforeach(module_name)
endforeach(metavision_offer)
set(METAVISION_SDK_MODULES_AVAILABLE ${METAVISION_SDK_OPEN_MODULES_AVAILABLE} ${METAVISION_SDK_ESSENTIALS_MODULES_AVAILABLE})

################################################### Find needed packages and programs

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# LibUSB
find_package(LibUSB REQUIRED)

# Boost
set (boost_components_to_find program_options filesystem timer chrono thread)
find_package(Boost COMPONENTS ${boost_components_to_find} REQUIRED)

# Bindings for python 3
if(COMPILE_PYTHON3_BINDINGS)
    if (MSVC)
        set(PYBIND11_CPP_STANDARD /std:c++${CMAKE_CXX_STANDARD})
    else ()
        set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD})
    endif ()

    find_package(pybind11 REQUIRED)
endif(COMPILE_PYTHON3_BINDINGS)

# OpenCV
find_package(OpenCV COMPONENTS core highgui imgproc videoio imgcodecs calib3d objdetect REQUIRED)

# Doxygen documentation
if (GENERATE_DOC)
    include(documentation)
endif (GENERATE_DOC)

# Tests
include(CTest)
if (BUILD_TESTING)
    # Gtest
    find_package(GTest REQUIRED)

    # Pytests
    if(NOT Python3_EXECUTABLE)
        set(Python3_FIND_FRAMEWORK LAST)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif(NOT Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE res
        ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if(res AND NOT res EQUAL 0)
        message(FATAL_ERROR "Error when executing '${Python3_EXECUTABLE} -c \"import pytest\"'"
        "\n${err}Either install pytest or disable option BUILD_TESTING\n"
        "To install pytest, run :\n    ${Python3_EXECUTABLE} -m pip install pytest")
    endif(res AND NOT res EQUAL 0)

    # If compiling the python bindings, extract the version of the python bindings tested
    if(COMPILE_PYTHON3_BINDINGS)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import platform; print(platform.python_version())"
            RESULT_VARIABLE res
            ERROR_VARIABLE err
            OUTPUT_VARIABLE PYTHON_VERSION_TESTED
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(res AND NOT res EQUAL 0)
            message(FATAL_ERROR "Error when executing '${Python3_EXECUTABLE} -c \"import platform; print(platform.python_version())\""
            "\n${err}")
        endif(res AND NOT res EQUAL 0)
    endif(COMPILE_PYTHON3_BINDINGS)

    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/datasets" AND GIT_LFS_NOT_AVAILABLE)
        message("----------------------------------------------------------------------------------------------")
        message("|  To run tests, please download https://dataset.prophesee.ai/index.php/s/ozjYOAAKTUshudQ    |")
        message("|  and extract its content in <METAVISION_FOLDER>/datasets/                                  |")
        message("|  Warning : the whole test archive weighs about 3 Gb.                                       |")
        message("----------------------------------------------------------------------------------------------\n")
    endif()

    set(PYTEST_CMD ${Python3_EXECUTABLE} -m pytest)
    lfs_download(datasets/gen31_timer.raw)
    lfs_download(datasets/gen4_evt2_hand.raw)
    lfs_download(datasets/gen4_evt3_hand.raw)
    if(COMPILE_PYTHON3_BINDINGS)
        lfs_download(datasets/gen4_evt2_hand_cd.dat)
        # python modules pytest data
        lfs_download("datasets/metavision_core/")
    endif(COMPILE_PYTHON3_BINDINGS)

    function(add_test_app app_name)
        string(REPLACE "_" "-" test_name "${app_name}")
        string(APPEND test_name "-test")

        # If when calling the app you pass the python script to run, then use it. If not,
        # assume that the python script to run is called ${app_name}_pytest.py and is in
        # the directory from which you are calling the function
        if(ARGV1)
            set(pytest_file "${ARGV1}")
        else()
            set(pytest_file "${CMAKE_CURRENT_LIST_DIR}/${app_name}_pytest.py")
        endif(ARGV1)

        # If the pytest file is not in a subdirectory of the source directory (which can happen if for example
        # the script is the result of a "configure_file" with destination outside the source tree, then we need to copy
        # the ini and conftest pytest files:
        if(NOT pytest_file MATCHES "^${PROJECT_SOURCE_DIR}")
            get_filename_component(pytest_file_dir "${pytest_file}" DIRECTORY)
            file(COPY "${PROJECT_SOURCE_DIR}/conftest.py" DESTINATION "${pytest_file_dir}")
            file(COPY "${PROJECT_SOURCE_DIR}/pytest.ini" DESTINATION "${pytest_file_dir}")
        endif(NOT pytest_file MATCHES "^${PROJECT_SOURCE_DIR}")

        if (NOT EXISTS "${pytest_file}")
            message(FATAL_ERROR "ERROR : cannot add test for app ${app_name} because file ${pytest_file} does not exist")
        endif(NOT EXISTS "${pytest_file}")
        add_test(
            NAME ${test_name}
            COMMAND ${PYTEST_CMD} ${pytest_file} -vv --capture=no --color=yes --junitxml=${JUNIT_XML_OUTPUT_DIRECTORY}/${test_name}.xml
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${app_name}>
        )
        set_property(TEST ${test_name} PROPERTY ENVIRONMENT
            PYTHONPATH=${PROJECT_SOURCE_DIR}/utils/python # to be able to use prophesee_utils python module
            MV_HAL_PLUGIN_PATH=${HAL_BUILD_PLUGIN_PATH}
        )
    endfunction(add_test_app)

    # Paths where gtest and pytests xml output files are stored
    set(TEST_XML_OUTPUT_DIRECTORY "${GENERATE_FILES_DIRECTORY}/tests_xml_output")
    set(JUNIT_XML_OUTPUT_DIRECTORY "${TEST_XML_OUTPUT_DIRECTORY}/junit/")
    set(GTEST_XML_OUTPUT_DIRECTORY "${TEST_XML_OUTPUT_DIRECTORY}/gtest/")

    include(register_gtest)

endif (BUILD_TESTING)

# Code coverage
if (CODE_COVERAGE)
    include(code_coverage)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_COVERAGE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_COVERAGE}")

    set(CODE_COVERAGE_OUTPUT_DIR ${GENERATE_FILES_DIRECTORY}/code_coverage)
    file(MAKE_DIRECTORY "${CODE_COVERAGE_OUTPUT_DIR}")
endif(CODE_COVERAGE)

################################################### COMPILATION

# Add custom targets and function :
include(uninstall)
include(add_library_version_header)
include(add_cpack_component)
include(pybind11_create_module)
include(create_metavision_open_archive)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/custom_targets_metavision_professional")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/custom_targets_metavision_professional)
    include(create_metavision_professional_archives)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/custom_targets_metavision_professional")

# Add Metavision Open debian packages :
add_cpack_component(PUBLIC metavision-open metavision-open-bin metavision-open-dev metavision-open-samples)
if(COMPILE_PYTHON3_BINDINGS)
    add_cpack_component(PUBLIC metavision-open-python metavision-open-python-samples)
endif(COMPILE_PYTHON3_BINDINGS)

# Add Metavision Essentials debian packages :
if (METAVISION_SDK_ESSENTIALS_MODULES_AVAILABLE)
    add_cpack_component(PUBLIC metavision-essentials-licence metavision-essentials metavision-essentials-bin)
    if(COMPILE_PYTHON3_BINDINGS)
        add_cpack_component(PUBLIC metavision-essentials-python metavision-essentials-python-samples)
    endif(COMPILE_PYTHON3_BINDINGS)

    if(EXISTS "${PROJECT_SOURCE_DIR}/licensing/LICENSE_ESSENTIALS")
        install(FILES ${PROJECT_SOURCE_DIR}/licensing/LICENSE_ESSENTIALS
                DESTINATION share/metavision/licensing
                COMPONENT metavision-essentials-licence)
    endif()
endif (METAVISION_SDK_ESSENTIALS_MODULES_AVAILABLE)

################################
#            Utils            ##
################################
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/utils/CMakeLists.txt")
    add_subdirectory(utils)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/utils/CMakeLists.txt")

################################
#             HAL             ##
################################
add_subdirectory(hal)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/hal_psee_plugins/CMakeLists.txt")
    add_subdirectory(hal_psee_plugins)
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/hal_psee_plugins/CMakeLists.txt")

################################
#             SDK             ##
################################
add_subdirectory(sdk)

################################
#           Designer          ##
################################
if (COMPILE_PYTHON3_BINDINGS AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/designer/CMakeLists.txt")
    add_subdirectory(designer)
endif(COMPILE_PYTHON3_BINDINGS AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/designer/CMakeLists.txt")

################################
#      Standalone samples     ##
################################
add_subdirectory(standalone_samples)

################################
#           Documentation     ##
################################
if (GENERATE_DOC)
    add_subdirectory(doc)
endif (GENERATE_DOC)

################################################### CPack for debian packages

include(deb_packages)
