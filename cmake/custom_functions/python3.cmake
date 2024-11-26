# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

# Helper function to make a "global alias" library : alias library have the same scope
# as the aliased library, so if the library is IMPORTED not GLOBAL, the alias library won't be GLOBAL either
# anyway, alias library are not supported for cmake < 3.10.3, so we need a workaround
function(_add_global_alias_library lib alias_lib)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE _CMAKE_PROPERTY_LIST)
    string(REGEX REPLACE ";" "\\\\;" _CMAKE_PROPERTY_LIST "${_CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" _CMAKE_PROPERTY_LIST "${_CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES _CMAKE_PROPERTY_LIST)
    if (NOT TARGET ${alias_lib})
        add_library(${alias_lib} SHARED IMPORTED GLOBAL)
        foreach (_prop ${_CMAKE_PROPERTY_LIST})
            if (NOT "${_prop}" MATCHES "^(NAME|TYPE)$")
                set(_props "")
                if ("${_prop}" MATCHES "<CONFIG>")
                    foreach (_build_type DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
                        string(REPLACE "<CONFIG>" "${_build_type}" _prop_tmp "${_prop}")
                        list(APPEND _props "${_prop_tmp}")
                    endforeach()
                else()
                    set(_props "${_prop}")
                endif()
                foreach (_prop IN LISTS _props)
                    get_target_property(_propval ${lib} "${_prop}")
                    if (_propval)
                        set_target_properties(${alias_lib} PROPERTIES "${_prop}" "${_propval}")
                    endif()
                endforeach()
            endif()
        endforeach()
        if (MSVC)
            # workaround for a bug in Python on Windows, where the debug lib links to the relase lib file with a relative path
            set_property(TARGET ${alias_lib} APPEND PROPERTY INTERFACE_LINK_DIRECTORIES "$<$<CONFIG:Debug>:$<TARGET_LINKER_FILE_DIR:${alias_lib}>>")
        endif()
    endif()
endfunction()

set(_SAVE_CMAKE_FIND_USE_CMAKE_PATH ${CMAKE_FIND_USE_CMAKE_PATH})
if(MSVC)
    # this is needed to make sure we don't use vcpkg cmake wrapper for python
    set(CMAKE_FIND_USE_CMAKE_PATH FALSE)
endif()
set(Python3_FIND_FRAMEWORK LAST)

if (COMPILE_PYTHON3_BINDINGS)
    if(PYBIND11_PYTHON_VERSIONS)
        file(MAKE_DIRECTORY ${GENERATE_FILES_DIRECTORY}/python3)
        file(WRITE ${GENERATE_FILES_DIRECTORY}/python3/CMakeLists.txt "
            find_package(Python3 \${_python_version} EXACT COMPONENTS Interpreter Development REQUIRED)
            set(PYTHON_\${_python_version}_EXECUTABLE \${Python3_EXECUTABLE} CACHE PATH \"\")
            _add_global_alias_library(Python3::Python Python3::Python_\${_python_version})
        ")
        foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
            # FindPython3.cmake stores internal variables relative to the directory in which it is called
            # If we want to call it multiple times for different python versions, it must be called from
            # different folders (see https://gitlab.kitware.com/cmake/cmake/-/issues/21797)
            add_subdirectory(${GENERATE_FILES_DIRECTORY}/python3 ${PROJECT_BINARY_DIR}/cmake/python3/${_python_version})
        endforeach()
    else()
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        string(CONCAT _short_python3_version "${Python3_VERSION_MAJOR}" "." "${Python3_VERSION_MINOR}")
        set(PYBIND11_PYTHON_VERSIONS "${_short_python3_version}")
        set(PYTHON_${_short_python3_version}_EXECUTABLE ${Python3_EXECUTABLE} CACHE PATH "")
        _add_global_alias_library(Python3::Python Python3::Python_${_short_python3_version})
    endif()
    foreach (_python_version ${PYBIND11_PYTHON_VERSIONS})
        # this is the extension we need to set for the python bindings module
        if(CMAKE_CROSSCOMPILING)
            # When cross compiling, we cannot run the python interpreter as it might be compiled for a different architecture.
            # Therefore we ask for the user to have already set the library extension suffix.
            if(NOT DEFINED PYTHON_${_python_version}_MODULE_EXTENSION)
                message(FATAL_ERROR "CMake variable 'PYTHON_${_python_version}_MODULE_EXTENSION' needs to be defined. "
                                    "One can run '$ python3-config --extension-suffix' on your targeted platform to get the actual value (eg. '.cpython-36m-x86_64-linux-gnu.so').
                ")
            endif()
        else()
            execute_process(
                COMMAND "${PYTHON_${_python_version}_EXECUTABLE}" "-c"
                        "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'));"
                OUTPUT_VARIABLE PYTHON_${_python_version}_MODULE_EXTENSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif(CMAKE_CROSSCOMPILING)

        # this is the path where we install our python modules for cpack DEB (system) packages ...
        execute_process(
            COMMAND "${PYTHON_${_python_version}_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/utils/scripts/get_prefered_site_packages.py" "--site-type" "system"
            OUTPUT_VARIABLE PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        file(TO_CMAKE_PATH "${PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES}" PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES)
        # ... it must be relative
        string(REGEX REPLACE "^${STAGING_DIR_NATIVE}/usr/" "" PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES "${PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES}")

        # this is the path where we install our python modules in the local system install tree
        if (PYTHON3_SITE_PACKAGES)
            # If a PYTHON3_SITE_PACKAGES variable was provided use it
            set(PYTHON_${_python_version}_LOCAL_SITE_PACKAGES ${PYTHON3_SITE_PACKAGES})
            file(TO_CMAKE_PATH "${PYTHON_${_python_version}_LOCAL_SITE_PACKAGES}" PYTHON_${_python_version}_LOCAL_SITE_PACKAGES)
        else (PYTHON3_SITE_PACKAGES)
            # Otherwise, if no module path is provided, find a proper path to install our modules
            # note : we do not use the PYTHON_SITE_PACKAGES found by pybind11, as it is wrong
            execute_process(
                COMMAND "${PYTHON_${_python_version}_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/utils/scripts/get_prefered_site_packages.py" "--site-type" "local" "--prefer-pattern" "${CMAKE_INSTALL_PREFIX}"
                OUTPUT_VARIABLE PYTHON_${_python_version}_LOCAL_SITE_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            file(TO_CMAKE_PATH "${PYTHON_${_python_version}_LOCAL_SITE_PACKAGES}" PYTHON_${_python_version}_LOCAL_SITE_PACKAGES)
        endif (PYTHON3_SITE_PACKAGES)
        # ... it must be relative
        string(REGEX REPLACE "^${STAGING_DIR_NATIVE}/usr/(local/)?" "" PYTHON_${_python_version}_LOCAL_SITE_PACKAGES "${PYTHON_${_python_version}_LOCAL_SITE_PACKAGES}")
    endforeach ()

    # this variable is used to create all python versions packages variables for cpack
    # but not all of them will be generated, only the one indicated by PYBIND11_PYTHON_VERSIONS
    set (PYTHON3_ALL_VERSIONS "3.7;3.8;3.9;3.10;3.11;3.12")

    # this variable is used to set the default version for package dependency, i.e this version
    # is always available for the current installation
    if (UNIX AND NOT APPLE AND (NOT DEFINED PYTHON3_DEFAULT_VERSION))
        set (PYTHON3_DEFAULT_VERSION "3.9")
        find_program(_lsb_release_exec lsb_release)
        if (_lsb_release_exec)
            execute_process(COMMAND ${_lsb_release_exec} -cs
                OUTPUT_VARIABLE _ubuntu_platform
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if ("${_ubuntu_platform}" STREQUAL "jammy")
              set (PYTHON3_DEFAULT_VERSION "3.10")
            elseif ("${_ubuntu_platform}" STREQUAL "noble")
              set (PYTHON3_DEFAULT_VERSION "3.12")
            endif ()
        endif()
    else()
        # arbitrarily chose the first one
        list(GET PYBIND11_PYTHON_VERSIONS 0 PYTHON3_DEFAULT_VERSION)
    endif()
endif()

if (BUILD_TESTING)
    # TODO : MV-167, remove this part, there should be no more PYTEST_PYTHON_VERSION variable, and we
    # should test that pytest is available for all versions in the foreach loop above
    if (PYTEST_PYTHON_VERSION)
        find_package(Python3 ${PYTEST_PYTHON_VERSION} EXACT COMPONENTS Interpreter REQUIRED)
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(PYTEST_PYTHON_VERSION ${Python3_VERSION})
    endif()
    if(NOT CMAKE_CROSSCOMPILING) # When cross compiling for a different architecture, target's binary aren't compatible and cannot be run.
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
    endif(NOT CMAKE_CROSSCOMPILING)
    set(PYTHON_${PYTEST_PYTHON_VERSION}_EXECUTABLE ${Python3_EXECUTABLE} CACHE PATH "")
endif ()

if(MSVC)
    # restore saved value
    set(CMAKE_FIND_USE_CMAKE_PATH ${_SAVE_CMAKE_FIND_USE_CMAKE_PATH})
endif()

if (COMPILE_PYTHON3_BINDINGS)
    # pybind11 should not look for Python anymore, we already handled it
    set(PYBIND11_NOPYTHON On)
    find_package(pybind11 2.7 REQUIRED)

    # private helper function to install a python module for a specific python version with LOCAL and/or SYSTEM python component
    function(_install_python_bindings_for_version target_name python_version component)
        set (component "${component}-python${python_version}")
        cmake_parse_arguments(PARSE_ARGV 3 ARG "LOCAL;SYSTEM" "" "")

        if (ARG_SYSTEM OR NOT ARG_LOCAL)
            install(TARGETS ${target_name}
                RUNTIME
                    DESTINATION "${PYTHON_${python_version}_SYSTEM_SITE_PACKAGES}"
                    COMPONENT ${component} EXCLUDE_FROM_ALL
                LIBRARY
                    DESTINATION "${PYTHON_${python_version}_SYSTEM_SITE_PACKAGES}"
                    COMPONENT ${component} EXCLUDE_FROM_ALL
                ARCHIVE
                    DESTINATION "${PYTHON_${python_version}_SYSTEM_SITE_PACKAGES}"
                    COMPONENT ${component} EXCLUDE_FROM_ALL
            )
        endif()

        if (ARG_LOCAL OR NOT ARG_SYSTEM)
            install(TARGETS ${target_name}
                RUNTIME
                    DESTINATION "${PYTHON_${python_version}_LOCAL_SITE_PACKAGES}"
                    COMPONENT ${component}-local-install
                LIBRARY
                    DESTINATION "${PYTHON_${python_version}_LOCAL_SITE_PACKAGES}"
                    COMPONENT ${component}-local-install
                ARCHIVE
                    DESTINATION "${PYTHON_${python_version}_LOCAL_SITE_PACKAGES}"
                    COMPONENT ${component}-local-install
            )
        endif()
    endfunction()

    # private helper function to add a pybind11 module for a specific python version
    function(_add_pybind11_module_for_version target_name python_version output_name)
        cmake_parse_arguments(PARSE_ARGV 3 ARG
        "STATIC;SHARED;MODULE;THIN_LTO;OPT_SIZE;NO_EXTRAS;WITHOUT_SOABI" "" "")

        if(ARG_STATIC)
            set(lib_type STATIC)
        elseif(ARG_SHARED)
            set(lib_type SHARED)
        else()
            set(lib_type MODULE)
        endif()

        add_library(${target_name} ${lib_type} ${ARG_UNPARSED_ARGUMENTS})

        if (NOT APPLE)
            target_link_libraries(${target_name} PRIVATE pybind11::module Python3::Python_${python_version})
        else()
            get_target_property(_python_include_dirs Python3::Python_${python_version} INTERFACE_INCLUDE_DIRECTORIES)
            target_include_directories(${target_name} PRIVATE ${_python_include_dirs})
            target_link_libraries(${target_name} PRIVATE pybind11::module)
            target_link_options(${target_name} PRIVATE -undefined dynamic_lookup)
        endif()

        if(lib_type STREQUAL "MODULE")
            target_link_libraries(${target_name} PRIVATE pybind11::module)
        else()
            target_link_libraries(${target_name} PRIVATE pybind11::embed)
        endif()

        if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
            set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
        endif()

        if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
            set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
        endif()

        # If we don't pass a WITH_SOABI or WITHOUT_SOABI, use our own default handling of extensions
        if(NOT ARG_WITHOUT_SOABI OR NOT "WITH_SOABI" IN_LIST ARG_UNPARSED_ARGUMENTS)
            set_target_properties(${target_name} PROPERTIES PREFIX "" SUFFIX "${PYTHON_${python_version}_MODULE_EXTENSION}")
        endif()

        if(ARG_NO_EXTRAS)
            return()
        endif()

        if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
            if(ARG_THIN_LTO)
                target_link_libraries(${target_name} PRIVATE pybind11::thin_lto)
            else()
                target_link_libraries(${target_name} PRIVATE pybind11::lto)
            endif()
        endif()

        if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
            # Strip unnecessary sections of the binary on Linux/macOS
            pybind11_strip(${target_name})
        endif()

        if(MSVC)
            target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
        endif()

        if(ARG_OPT_SIZE)
            target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
        endif()

        set_target_properties(${target_name}
            PROPERTIES
                OUTPUT_NAME "${output_name}"
                ARCHIVE_OUTPUT_NAME "${target_name}"
                PDB_NAME "${target_name}"
                COMPILE_PDB_NAME "${target_name}"
                LIBRARY_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
                ARCHIVE_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
                RUNTIME_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
        )
    endfunction()

    # private helper function
    function(_add_python_bindings module libname component)
        set(multiValueArgs SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILE_DEFINITIONS DEPENDENCIES)
        cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

        if(WIN32 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
            file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                DESTINATION ${PROJECT_BINARY_DIR}/py3/${CMAKE_BUILD_TYPE}/${module}
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
                install(DIRECTORY ${PROJECT_BINARY_DIR}/py3/${CMAKE_BUILD_TYPE}/${module}
                        DESTINATION "${PYTHON_${_python_version}_LOCAL_SITE_PACKAGES}"
                        COMPONENT ${component}-python${_python_version}-local-install
                        PATTERN __pycache__ EXCLUDE)
            endforeach()
        endif()

        set(target ${libname}_python3)
        add_custom_target(${target})
        foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
            set(_target "${target}_${_python_version}")
            _add_pybind11_module_for_version(${_target} ${_python_version} ${libname} MODULE)

            if(ARG_SOURCES)
                target_sources(${_target} PRIVATE ${ARG_SOURCES})
            endif()

            if(ARG_INCLUDE_DIRECTORIES)
                cmake_parse_arguments(_include_directories "" "" "PUBLIC;INTERFACE;PRIVATE" ${ARG_INCLUDE_DIRECTORIES})
                foreach(_level PUBLIC INTERFACE PRIVATE)
                    foreach(_dir IN LISTS _include_directories_${_level})
                        target_include_directories(${_target} ${_level} ${_dir})
                    endforeach()
                endforeach()
            endif()

            if(ARG_LINK_LIBRARIES)
                cmake_parse_arguments(_link_libraries "" "" "PUBLIC;INTERFACE;PRIVATE" ${ARG_LINK_LIBRARIES})
                foreach(_level PUBLIC INTERFACE PRIVATE)
                    foreach(_lib IN LISTS _link_libraries_${_level})
                        target_link_libraries(${_target} ${_level} ${_lib})
                    endforeach()
                endforeach()
            endif()

            if(ARG_COMPILE_DEFINITIONS)
                cmake_parse_arguments(_compile_definitions "" "" "PUBLIC;INTERFACE;PRIVATE" ${ARG_COMPILE_DEFINITIONS})
                foreach(_level PUBLIC INTERFACE PRIVATE)
                    foreach(_def IN LISTS _compile_definitions_${_level})
                        target_compile_definitions(${_target} ${_level} ${_def})
                    endforeach()
                endforeach()
            endif()

            if(ARG_DEPENDENCIES)
                foreach(_dep ${ARG_DEPENDENCIES})
                    add_dependencies(${_target} ${_dep})
                endforeach()
            endif()

            _install_python_bindings_for_version(${_target} ${_python_version} ${component})

            add_dependencies(${target} ${_target})
        endforeach()
    endfunction()


    #############################
    # HAL specific functions
    #############################

    # function to add python bindings for HAL for all python versions
    function(add_hal_python_bindings)
        set(module "metavision_hal")
        if(WIN32)
            set(libname ${module}_internal)
        else()
            set(libname ${module})
        endif()
        set(component "${libname}")
        string(REPLACE "_" "-" component ${component})
        _add_python_bindings(${module} ${libname} ${component} ${ARGV})
    endfunction()


    #############################
    # SDK specific functions
    #############################

    # function to install a directory with LOCAL and/or SYSTEM python component
    function(install_sdk_python_module module directory)
        foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
            cmake_parse_arguments(ARG "LOCAL;SYSTEM" "" "EXCLUDED_PATTERNS" ${ARGN})

            set(_excluded_patterns PATTERN "__pycache__" EXCLUDE)
            if (ARG_EXCLUDED_PATTERNS)
                foreach (_pattern ${ARG_EXCLUDED_PATTERNS})
                    list(APPEND _excluded_patterns PATTERN ${_pattern} EXCLUDE)
                endforeach()
            endif ()

            set (component "metavision-sdk-${module}-python")
            if (ARG_SYSTEM OR NOT ARG_LOCAL)
                install(
                    DIRECTORY "${directory}"
                    DESTINATION "${PYTHON_${_python_version}_SYSTEM_SITE_PACKAGES}"
                    COMPONENT "${component}" EXCLUDE_FROM_ALL
                    ${_excluded_patterns}
                )
            endif()

            if (ARG_LOCAL OR NOT ARG_SYSTEM)
                install(
                    DIRECTORY "${directory}"
                    DESTINATION "${PYTHON_${_python_version}_LOCAL_SITE_PACKAGES}"
                    COMPONENT "${component}-local-install"
                    ${_excluded_patterns}
                )
            endif()
        endforeach(_python_version ${PYBIND11_PYTHON_VERSIONS})
    endfunction()

    # function to add a python bindings for a SDK module for all python versions
    function(add_sdk_python_bindings module)
        set(module "metavision_sdk_${module}")
        if(WIN32)
            set(libname ${module}_internal)
        else()
            set(libname ${module})
        endif()
        set(component "${libname}")
        string(REPLACE "_" "-" component ${component})
        _add_python_bindings(${module} ${libname} ${component} ${ARGV})
    endfunction()

    #####################################################################
    #
    # Adds python (based on binary package) cpack component(s)
    #
    #
    # usage :
    #     add_python_cpack_components(
    #         <PRIVATE|PUBLIC> <components>...
    #        [<PRIVATE|PUBLIC> <components>...]
    #      )
    #
    #
    #  Given a list of components, add the corresponding python versions of the components according to the value of the
    #  PYBIND11_PYTHON_VERSIONS variable to the list of debian packages to generate. Both PRIVATE and PUBLIC components will be
    #  created when doing "cpack -G DEB", while only the PUBLIC component will be created when building target
    #  "public_deb_packages"
    #
    #
    function(add_python_cpack_components)
        cmake_parse_arguments(CPACK_COMPONENTS "" "" "PUBLIC;PRIVATE" ${ARGN})

        # Get current list of cpack components
        get_property(python_components_to_install_internal_tmp GLOBAL PROPERTY list_python_cpack_internal_components)
        get_property(python_components_to_install_public_tmp GLOBAL PROPERTY list_python_cpack_public_components)
        get_property(components_to_install_internal_tmp GLOBAL PROPERTY list_cpack_internal_components)
        get_property(components_to_install_public_tmp GLOBAL PROPERTY list_cpack_public_components)

        # Iterate through the input arguments to add them in the list of cpack components
        foreach(compon IN LISTS CPACK_COMPONENTS_PUBLIC)
            if (${compon} IN_LIST python_components_to_install_internal_tmp)
                message(SEND_ERROR
                "Error when calling function add_python_cpack_components : component ${compon} has already been listed as private, you cannot add it as public as well")
                return()
            endif()
            if (NOT ${compon} IN_LIST python_components_to_install_public_tmp)
                set (python_components_to_install_public_tmp ${python_components_to_install_public_tmp} ${compon})
            endif ()
        endforeach(compon)
        foreach(compon IN LISTS CPACK_COMPONENTS_PRIVATE)
            if (${compon} IN_LIST python_components_to_install_public_tmp)
                message(SEND_ERROR
                "Error when calling function add_python_cpack_components : component ${compon} has already been listed as public, you cannot add it as private as well")
                return()
            endif()
            if (NOT ${compon} IN_LIST python_components_to_install_internal_tmp)
                set (python_components_to_install_internal_tmp ${python_components_to_install_internal_tmp} ${compon})
            endif ()
        endforeach(compon)

        # Fix the list of packages for each python version
        foreach(compon IN LISTS python_components_to_install_internal_tmp)
            foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
                set (components_to_install_internal_tmp ${components_to_install_internal_tmp} ${compon}-python${_python_version})
            endforeach()
        endforeach()
        foreach(compon IN LISTS python_components_to_install_public_tmp)
            foreach(_python_version ${PYBIND11_PYTHON_VERSIONS})
                set (components_to_install_public_tmp ${components_to_install_public_tmp} ${compon}-python${_python_version})
            endforeach()
        endforeach()

        # Set back global variables
        set_property(GLOBAL PROPERTY list_python_cpack_internal_components "${python_components_to_install_internal_tmp}")
        set_property(GLOBAL PROPERTY list_python_cpack_public_components "${python_components_to_install_public_tmp}")
        set_property(GLOBAL PROPERTY list_cpack_internal_components "${components_to_install_internal_tmp}")
        set_property(GLOBAL PROPERTY list_cpack_public_components "${components_to_install_public_tmp}")

    endfunction(add_python_cpack_components)

    #####################################################################
    #
    # Return path to cpython libraries to be added to PYTHONPATH.
    #
    # usage :
    #     get_pybind_pythonpath(PYBIND_PATH)
    #     # Add ${PYBIND_PATH} to PYTHONPATH
    #
    function(get_pybind_pythonpath output_var)
        if(WIN32)
            set(pythonpath_value "${PYTHON3_OUTPUT_DIR}/$<CONFIG>")
        else()
            set(pythonpath_value "${PYTHON3_OUTPUT_DIR}")
        endif()
        set(${output_var} "${pythonpath_value}" PARENT_SCOPE)
    endfunction(get_pybind_pythonpath)
endif()
