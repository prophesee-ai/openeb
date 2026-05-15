#[=======================================================================[.rst:
LibUSB
--------

Find the libusb includes and library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``libusb-1.0``, if
libusb-1.0 has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

::

  LIBUSB_INCLUDE_DIR    - where to find libusb.h
  LIBUSB_LIBRARIES      - List of libraries when using libusb.
  LIBUSB_FOUND          - True if libusb found.

#]=======================================================================]


# Clear the cache
unset (LIBUSB_INCLUDE_DIR CACHE)
unset (LIBUSB_LIBRARIES CACHE)

if(MSVC)
    # Typically on MSVC platforms there is no PkgConfig, so we rely on some defaults.
    # When using VCPKG for example, it will fix the path searching automatically.
    find_path(LIBUSB_INCLUDE_DIR 
        NAMES libusb.h
        PATH_SUFFIXES libusb-1.0
    )

    find_library(LIBUSB_LIBRARY
        NAMES libusb-1.0
    )
else()
    # use pkg-config to get the directories and then use these values
    # in the FIND_PATH() and FIND_LIBRARY() calls

    find_package(PkgConfig)
    pkg_check_modules(PC_LIBUSB libusb-1.0)

    find_path(LIBUSB_INCLUDE_DIR 
        NAMES libusb.h
        PATHS ${PC_LIBUSB_INCLUDEDIR} ${PC_LIBUSB_INCLUDE_DIRS}
    )

    find_library(LIBUSB_LIBRARY NAMES usb-1.0
        PATHS ${PC_LIBUSB_LIBDIR} ${PC_LIBUSB_LIBRARY_DIRS}
    )        
    
endif(MSVC)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibUSB DEFAULT_MSG LIBUSB_INCLUDE_DIR LIBUSB_LIBRARY)

if (LIBUSB_FOUND AND NOT TARGET libusb-1.0)
    add_library(libusb-1.0 UNKNOWN IMPORTED)
    set_target_properties(libusb-1.0 PROPERTIES 
                            IMPORTED_LOCATION ${LIBUSB_LIBRARY}
                            INTERFACE_INCLUDE_DIRECTORIES ${LIBUSB_INCLUDE_DIR})
endif (LIBUSB_FOUND AND NOT TARGET libusb-1.0)

mark_as_advanced(
    LIBUSB_INCLUDE_DIR
    LIBUSB_LIBRARY
)

set (LIBUSB_LIBRARIES ${LIBUSB_LIBRARY})