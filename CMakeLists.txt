# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    # We have to set the build type BEFORE `project()` is called. Otherwise, a custom toolchain may
    # set the variable for us (as it is done on Windows)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

option(BUILD_TESTING "Build test suites" OFF)

cmake_minimum_required(VERSION 3.5)

project(metavision VERSION 5.1.1)
set(PROJECT_VERSION_SUFFIX "")

if(PROJECT_VERSION_SUFFIX STREQUAL "")
    set(PROJECT_VERSION_FULL "${PROJECT_VERSION}")
else(PROJECT_VERSION_SUFFIX STREQUAL "")
    set(PROJECT_VERSION_FULL "${PROJECT_VERSION}-${PROJECT_VERSION_SUFFIX}")
endif(PROJECT_VERSION_SUFFIX STREQUAL "")

include(CTest)

# Set output directory for build targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/ CACHE PATH "Output directory of libraries.")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/ CACHE PATH "Output directory of all executables.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/ CACHE PATH "Output directory of all executables.")
set(PYTHON3_OUTPUT_DIR ${PROJECT_BINARY_DIR}/py3/ CACHE PATH "Output directory of python 3 libraries.")

# Set Platform Target name
set(TARGET_PLATFORM "" CACHE STRING "Optional targeted platform name for cross-compiling")

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
    set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/metavision-${PROJECT_VERSION_FULL}")
else (ANDROID)
    set(LIBRARY_INSTALL_DEST "lib")
    set(ARCHIVE_INSTALL_DEST "lib")
    set(RUNTIME_INSTALL_DEST "bin")
    set(HAL_INSTALL_PLUGIN_RELATIVE_PATH "lib/metavision/hal/plugins")
endif (ANDROID)
set(HAL_BUILD_PLUGIN_PATH "${PROJECT_BINARY_DIR}/${HAL_INSTALL_PLUGIN_RELATIVE_PATH}")

# Set output directory for generated files
set(GENERATE_FILES_DIRECTORY ${PROJECT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATE_FILES_DIRECTORY})

# Update CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules) # For find_packages()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/custom_functions) # For defining custom cmake functions
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/custom_targets) # For defining custom cmake targets
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/cpack) # For cpack

include(CMakeDependentOption)
include(overridden_cmake_functions)
include(lfs_download)
include(add_pytest)
include(common_macros)

if (ANDROID)
    if (NOT ENV_CMAKE)
        # Those OVERLOAD_ functions are used to be able to bypass default Metavision
        # implementation of Android support to give a different own depending on
        # dependencies management
        set(ENV_CMAKE ${PROJECT_SOURCE_DIR}/cmake/android/env.cmake)
        set(PREBUILT_BASE_DIR utils/android)
        set(PREBUILT_FILENAME prebuilt-3rdparty.tar.gz)

        # Setup 3rd party prebuilt libraries dir
        set(ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR ${GENERATE_FILES_DIRECTORY}/android/3rdparty)
        if (NOT EXISTS ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})
            lfs_download(COMPILATION IMMEDIATE ${PREBUILT_BASE_DIR}/${PREBUILT_FILENAME})
            message(STATUS "Unpacking ${PREBUILT_BASE_DIR}/${PREBUILT_FILENAME} in ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR}")
            file(MAKE_DIRECTORY ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar -xf ${PROJECT_SOURCE_DIR}/${PREBUILT_BASE_DIR}/${PREBUILT_FILENAME}
                WORKING_DIRECTORY ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR}
            )
        endif (NOT EXISTS ${ANDROID_PREBUILT_3RDPARTY_EXTRACT_DIR})

        # Use Android env.cmake to find required dependencies
        set(ANDROID_PREBUILT_3RDPARTY_DIR ${GENERATE_FILES_DIRECTORY}/android/3rdparty/prebuilt)
        include(${ENV_CMAKE})
    endif()

    # Include the function required to build an Android APK
    include(add_android_app)
endif()

###################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif (CMAKE_CUDA_COMPILER)

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
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4244 /wd4267")

    # Export every symbol when building DLLs. This is similar to the behavior of GCC on Linux.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # Since vcpkg does not automatically include top level include path
    # and to be able to include some "system files", like dirent.h
    find_path(VCPKG_INCLUDE_DIR dirent.h)
    include_directories(${VCPKG_INCLUDE_DIR})
elseif (NOT ANDROID AND "${TARGET_PLATFORM}" STREQUAL "")
    # Needed when building with GCC < 9
    if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        link_libraries(stdc++fs)
    endif()

    # Platforms with their own toolchains should define their own compile flags
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror=type-limits -Werror=sign-compare -Werror=implicit-fallthrough -Werror=unused-function -Werror=unused-but-set-variable -Werror=dangling-else -Werror=sequence-point -Werror=uninitialized")
    else (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror=type-limits -Werror=sign-compare -Werror=implicit-fallthrough -Werror=unused-function -Werror=unused-but-set-variable -Werror=dangling-else -Werror=sequence-point -Werror=maybe-uninitialized")
    endif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
endif()

if (WIN32)
    # We need to use '\\' instead of '/' for subkeys, as those are not file paths
    set(METAVISION_SUBKEY "Software\\Prophesee")
    set(METAVISION_SUBKEY_INSTALL_PATH "INSTALL_PATH")

    # replace '\\' by '\\\\' to finally get '\\' in the generated header files later in the code
    string(REPLACE "\\" "\\\\" METAVISION_SUBKEY ${METAVISION_SUBKEY})
endif()


add_subdirectory_if_exists(utils/platforms) # Load platform specific cmake early to override necessary options

################################################### CMake options

option(BUILD_SAMPLES "Build Metavision apps & samples" ON)
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
set (boost_components_to_find program_options timer chrono thread)
find_package(Boost COMPONENTS ${boost_components_to_find} REQUIRED)
add_compile_definitions(BOOST_BIND_GLOBAL_PLACEHOLDERS) ## needed to get rid of warning `#pragma message: The practice of declaring the Bind placeholders (_1, _2, ...)`

# Python 3 for bindings and pytests
if (COMPILE_PYTHON3_BINDINGS OR BUILD_TESTING)
    include(python3)
endif ()

# OpenCV
find_package(OpenCV COMPONENTS core highgui imgproc videoio imgcodecs calib3d objdetect REQUIRED)

# HDF5
if (NOT ANDROID AND NOT HDF5_DISABLED)
    find_package(HDF5 1.10.2 COMPONENTS CXX)
    if (HDF5_FOUND AND WIN32)
        LIST(APPEND HDF5_DEFINITIONS H5_BUILT_AS_DYNAMIC_LIB)
    endif ()
endif ()

# Doxygen documentation
if (GENERATE_DOC)
    include(documentation)
endif (GENERATE_DOC)

# Tests
if (BUILD_TESTING)
    # Gtest
    find_package(GTest CONFIG)
    if (NOT GTest_FOUND)
        find_package(GTest REQUIRED)
    endif (NOT GTest_FOUND)
    if (NOT TARGET GTest::gmock)
        message(FATAL_ERROR
            "GMock can't be found. Please install 'libgmock-dev' package. "
            "Otherwise make sure it's compiled and installed along with your GoogleTest suite."
        )
    endif()

    # Pytests
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/datasets" AND GIT_LFS_NOT_AVAILABLE)
        message("--------------------------------------------------------------------------------------------------------------------------")
        message("|  Test data was not found. To run the tests, please download the test data adapted to your project:                     |")
        message("|  - OpenEB: https://kdrive.infomaniak.com/app/share/975517/2aa2545c-6b12-4478-992b-df2acfb81b38 (1.5 Gb archive)        |")
        message("|  - Metavision SDK: https://kdrive.infomaniak.com/app/share/975517/e1a64b9b-c7d3-4b7c-8f70-20bbf474c49d (5.4 Gb archive)|")
        message("--------------------------------------------------------------------------------------------------------------------------\n")
    endif()

    set(PYTEST_CMD ${PYTHON_${PYTEST_PYTHON_VERSION}_EXECUTABLE} -m pytest)

    # Perform additional test to ensure that variable `PYTEST_CMD` resulted to a valid executable.
    # Throw fatal error otherwise to inform user something went wrong.
    string(COMPARE EQUAL "${PYTEST_CMD}" "-m pytest" PYTEST_CMD_IS_INVALID)

    if(PYTEST_CMD_IS_INVALID)
        message(WARNING "Problem with finding valid 'pytest' version, try overriding with: '-DPYTEST_PYTHON_VERSION=<version>'! ")
    endif(PYTEST_CMD_IS_INVALID)

    if(CMAKE_CROSSCOMPILING)
        # When cross compiling, we cannot use python executable path from the Host
        set(PYTEST_CMD pytest) # TODO we need a proper way to find python executable path on the TARGET.
    endif(CMAKE_CROSSCOMPILING)


    lfs_download(datasets/openeb/gen31_timer.raw VALIDATION)
    lfs_download(datasets/openeb/gen31_timer.hdf5 VALIDATION)
    lfs_download(datasets/openeb/gen31_timer_holes.raw VALIDATION)
    lfs_download(datasets/openeb/gen31_timer_holes.hdf5 VALIDATION)
    lfs_download(datasets/openeb/gen4_evt2_hand.raw VALIDATION)
    lfs_download(datasets/openeb/gen4_evt2_hand.hdf5 VALIDATION)
    lfs_download(datasets/openeb/gen4_evt3_hand.raw VALIDATION)
    lfs_download(datasets/openeb/gen4_evt3_hand.hdf5 VALIDATION)
    lfs_download(datasets/openeb/gen4_evt4_hand.raw VALIDATION)
    lfs_download(datasets/openeb/blinking_gen4_with_ext_triggers.raw VALIDATION)
    lfs_download(datasets/openeb/blinking_gen4_with_ext_triggers.hdf5 VALIDATION)
    lfs_download(datasets/openeb/claque_doigt_evt21.raw VALIDATION)
    lfs_download(datasets/openeb/claque_doigt_evt21.hdf5 VALIDATION)
    lfs_download(datasets/openeb/claque_doigt_evt4.raw VALIDATION)
    lfs_download(datasets/openeb/standup_evt21-legacy.raw VALIDATION)
    lfs_download(datasets/openeb/standup_evt21-legacy.hdf5 VALIDATION)
    lfs_download(datasets/openeb/diff3d.raw VALIDATION)
    lfs_download(datasets/openeb/histo3d.raw VALIDATION)
    lfs_download(datasets/openeb/histo3d_padding.raw VALIDATION)
    lfs_download(datasets/openeb/test_start_after_0.dat VALIDATION)
    lfs_download(datasets/openeb/aer_8bits.raw VALIDATION)
    lfs_download(datasets/openeb/aer_4bits.raw VALIDATION)
    lfs_download(datasets/openeb/0101_cm_mtr12_output.raw VALIDATION)
    lfs_download(datasets/openeb/0101_cm_mtru_output.raw VALIDATION)
    lfs_download(datasets/openeb/lifo_evt21.raw VALIDATION)
    lfs_download(datasets/openeb/lifo_evt2.raw VALIDATION)
    lfs_download(datasets/openeb/lifo_evt3.raw VALIDATION)
    lfs_download(datasets/openeb/synced/recording_master.raw VALIDATION)
    lfs_download(datasets/openeb/synced/recording_slave_0.raw VALIDATION)
    lfs_download(datasets/openeb/synced/recording_slave_1.raw VALIDATION)
    if(COMPILE_PYTHON3_BINDINGS)
        lfs_download(datasets/openeb/blinking_leds.raw VALIDATION)
        lfs_download(datasets/openeb/gen4_evt2_hand_cd.dat VALIDATION)
        lfs_download(datasets/openeb/core VALIDATION)
    endif(COMPILE_PYTHON3_BINDINGS)


    function(add_test_app)
        cmake_parse_arguments(_TEST_APP_ARGS "" "SCRIPT_PATH" "ENV" ${ARGN})
        set(app_name "${_TEST_APP_ARGS_UNPARSED_ARGUMENTS}")
        string(REPLACE "_" "-" test_name "${app_name}")
        string(APPEND test_name "-test")

        # If when calling the app you pass the python script to run, then use it. If not,
        # assume that the python script to run is called ${app_name}_pytest.py and is in
        # the directory from which you are calling the function
        if (_TEST_APP_ARGS_SCRIPT_PATH)
            set(pytest_file "${_TEST_APP_ARGS_SCRIPT_PATH}")
        else ()
            set(pytest_file "${CMAKE_CURRENT_LIST_DIR}/${app_name}_pytest.py")
        endif (_TEST_APP_ARGS_SCRIPT_PATH)
        set (pytest_env "")
        if (_TEST_APP_ARGS_ENV)
            set(pytest_env "${_TEST_APP_ARGS_ENV}")
        endif (_TEST_APP_ARGS_ENV)

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

        # TODO : MV-167, put this in a foreach loop for each python versions
        add_pytest(NAME ${test_name} PATH ${pytest_file} WORKING_DIRECTORY $<TARGET_FILE_DIR:${app_name}> ENV ${pytest_env})

    endfunction(add_test_app)

    # Paths where gtest and pytests xml output files are stored
    set(TEST_OUTPUT_DIRECTORY "${GENERATE_FILES_DIRECTORY}/tests_output")
    set(TEST_XML_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIRECTORY}/xml")
    set(JUNIT_XML_OUTPUT_DIRECTORY "${TEST_XML_OUTPUT_DIRECTORY}/junit/")
    set(GTEST_XML_OUTPUT_DIRECTORY "${TEST_XML_OUTPUT_DIRECTORY}/gtest/")
    set(TEST_LOG_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIRECTORY}/log/")

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

################################################### Detect which SDK modules are available

set(METAVISION_SDK_MODULES_OPEN base core stream)
if (COMPILE_PYTHON3_BINDINGS)
    list(APPEND METAVISION_SDK_MODULES_OPEN core_ml)
endif (COMPILE_PYTHON3_BINDINGS)
set(METAVISION_SDK_MODULES_ADVANCED analytics cv ml)
if (NOT ANDROID)
    list(APPEND METAVISION_SDK_MODULES_OPEN ui)
    if(NOT DISABLE_METAVISION_SDK_MODULE_CALIBRATION_AND_CV3D)
        list(APPEND METAVISION_SDK_MODULES_ADVANCED calibration cv3d)
    endif()
endif (NOT ANDROID)

if(NOT EXISTS METAVISION_SELECTED_MODULES)
    set(METAVISION_SELECTED_MODULES "${METAVISION_SDK_MODULES_OPEN};${METAVISION_SDK_MODULES_ADVANCED}" CACHE STRING "Metavision SDK modules selected for compilation")
endif()

set(METAVISION_SDK_MODULES_AVAILABLE)
foreach(metavision_modules_set OPEN ADVANCED)
    set(METAVISION_SDK_${metavision_modules_set}_MODULES_AVAILABLE)
    message(STATUS "Building METAVISION_SDK_MODULES_${metavision_modules_set} modules : ${METAVISION_SDK_MODULES_${metavision_modules_set}}")
    foreach(module_name IN LISTS METAVISION_SDK_MODULES_${metavision_modules_set})
        if (${module_name} IN_LIST METAVISION_SELECTED_MODULES AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/sdk/modules/${module_name}/CMakeLists.txt")
            list(APPEND METAVISION_SDK_${metavision_modules_set}_MODULES_AVAILABLE "${module_name}")
        endif()
    endforeach()
    list(APPEND METAVISION_SDK_MODULES_AVAILABLE ${METAVISION_SDK_${metavision_modules_set}_MODULES_AVAILABLE})
endforeach()
message(STATUS "METAVISION_SELECTED_MODULES=${METAVISION_SELECTED_MODULES}")
message(STATUS "METAVISION_SDK_MODULES_AVAILABLE=${METAVISION_SDK_MODULES_AVAILABLE}")

################################################### COMPILATION

# Add custom targets and function :
include(uninstall)
include(add_library_version_header)
include(add_cpack_component)

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake/custom_targets_metavision_sdk")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/custom_targets_metavision_sdk)
    include(create_metavision_sdk_source_archives)
    include(create_metavision_sdk_ml_models_archive)
endif(EXISTS "${PROJECT_SOURCE_DIR}/cmake/custom_targets_metavision_sdk")

# Add Metavision Open debian packages :
add_cpack_component(PUBLIC metavision-openeb-lib metavision-openeb-bin metavision-openeb-dev metavision-openeb-samples)
add_cpack_component(PUBLIC metavision-openeb metavision-openeb-license)
if(COMPILE_PYTHON3_BINDINGS)
    add_cpack_component(PUBLIC metavision-openeb-python metavision-openeb-python-samples)
    add_python_cpack_components(PUBLIC metavision-openeb)

    install(FILES ${PROJECT_SOURCE_DIR}/utils/python/requirements_openeb.txt
            DESTINATION share/metavision/python_requirements
            COMPONENT metavision-openeb-python)
endif(COMPILE_PYTHON3_BINDINGS)

install(FILES
            ${PROJECT_SOURCE_DIR}/licensing/LICENSE_OPEN
            ${PROJECT_SOURCE_DIR}/licensing/OPEN_SOURCE_3RDPARTY_NOTICES
        DESTINATION share/metavision/licensing
        COMPONENT metavision-openeb-license)


# Add Metavision SDK debian packages :
if (METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)
    # Advanced
    add_cpack_component(PUBLIC metavision-sdk-advanced-lib metavision-sdk-advanced-bin metavision-sdk-advanced-dev metavision-sdk-advanced-samples metavision-sdk-advanced-license)
    if(COMPILE_PYTHON3_BINDINGS)
        add_cpack_component(PUBLIC metavision-sdk-advanced-python metavision-sdk-advanced-python-samples)
        add_python_cpack_components(PUBLIC metavision-sdk-advanced)

        install(FILES ${PROJECT_SOURCE_DIR}/utils/python/requirements_sdk_advanced.txt
                DESTINATION share/metavision/python_requirements
                COMPONENT metavision-sdk-advanced-python)
    endif(COMPILE_PYTHON3_BINDINGS)

    # SDK
    add_cpack_component(PUBLIC metavision-sdk-lib metavision-sdk-bin metavision-sdk-dev metavision-sdk-samples metavision-sdk)
    if(COMPILE_PYTHON3_BINDINGS)
        add_cpack_component(PUBLIC metavision-sdk-python metavision-sdk-python-samples)
        add_python_cpack_components(PUBLIC metavision-sdk)
    endif(COMPILE_PYTHON3_BINDINGS)

    if(EXISTS "${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK")
        install(FILES ${PROJECT_SOURCE_DIR}/licensing/LICENSE_METAVISION_SDK
                DESTINATION share/metavision/licensing
                COMPONENT metavision-sdk-advanced-license)
    endif()
endif (METAVISION_SDK_ADVANCED_MODULES_AVAILABLE)

################################
#            Utils            ##
################################
add_subdirectory_if_exists(utils)

################################
#             HAL             ##
################################
add_subdirectory(hal)
add_subdirectory_if_exists(hal_psee_plugins)

################################
#             SDK             ##
################################
add_subdirectory(sdk)

if (BUILD_SAMPLES)
    ################################
    #      Standalone samples     ##
    ################################
    add_subdirectory(standalone_samples)
    ################################
    #      Standalone apps        ##
    ################################
    add_subdirectory_if_exists(standalone_apps)
endif()
################################
#           Documentation     ##
################################
if (GENERATE_DOC)
    add_subdirectory(doc)
endif (GENERATE_DOC)

# Add script generation at the end, once all other subdirs have been processed
add_subdirectory(utils/scripts)

# Add open archive task after modules, hal and plugins that might affect it
include(create_metavision_open_archive)

# Create metavision-get-started archive
include(create_metavision_get_started_archive)

################################################### CPack for debian packages

include(deb_packages)

# Make sure meta-packages install an additional (dummy) file to avoid installation warnings
# with packages disappearing because their content is completely replaced
get_property(cpack_public_components GLOBAL PROPERTY list_cpack_public_components)
get_property(cpack_internal_components GLOBAL PROPERTY list_cpack_internal_components)
set(cpack_all_components ${cpack_public_components} ${cpack_internal_components})
list(REMOVE_DUPLICATES cpack_all_components)
file(WRITE ${GENERATE_FILES_DIRECTORY}/meta-package.log "")
foreach(comp ${cpack_all_components})
    if ("${comp}" MATCHES "^metavision-(openeb|sdk|sdk-advanced)-(lib|bin|dev|samples|python3.*|python|python-samples)$" OR
        "${comp}" STREQUAL "metavision-openeb" OR "${comp}" STREQUAL "metavision-sdk")
        install(FILES ${GENERATE_FILES_DIRECTORY}/meta-package.log
                DESTINATION share/metavision/log/${comp}
                COMPONENT ${comp}
        )
    endif()
endforeach()
