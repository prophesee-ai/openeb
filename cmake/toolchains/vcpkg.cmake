if (METAVISION_VCPKG_TOOLCHAIN)
    return()
endif()

if (VCPKG_DIRECTORY)
    set(ENV{_VCPKG_DIRECTORY} "${VCPKG_DIRECTORY}")
else()
    if (NOT DEFINED ENV{_VCPKG_DIRECTORY})
        message(FATAL_ERROR "VCPKG_DIRECTORY must be provided with this toolchain.")
    else()
        set(VCPKG_DIRECTORY "$ENV{_VCPKG_DIRECTORY}")
    endif()
endif()

file(TO_CMAKE_PATH "${VCPKG_DIRECTORY}" _vcpkg_directory)
set(_vcpkg_toolchain_file "${_vcpkg_directory}/scripts/buildsystems/vcpkg.cmake")
if(NOT EXISTS "${_vcpkg_toolchain_file}")
    message(FATAL_ERROR "VCPKG_DIRECTORY must point to a valid directory containing the VCPKG toolchain.")
endif()

set(METAVISION_VCPKG_TOOLCHAIN ON)
# do not use default overrided by vcpkg find_package, its wrapper prevents us from finding correct
# versions of python3 libraries, libusb targets, etc..
set(VCPKG_OVERRIDE_FIND_PACKAGE_NAME vcpkg_find_package)
include("${_vcpkg_toolchain_file}")
