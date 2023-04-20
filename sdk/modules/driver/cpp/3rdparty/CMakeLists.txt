if(HDF5_FOUND)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/hdf5_ecf/CMakeLists.txt")
        execute_process(
            COMMAND "${GIT_SCM}" submodule update --init -- "${CMAKE_CURRENT_LIST_DIR}/hdf5_ecf"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
    endif(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/hdf5_ecf/CMakeLists.txt")
    add_subdirectory(hdf5_ecf)

    # Note that we don't simply append the HDF5 ECF packages to the all/public components of cpack,
    # this would generate the DEB packages with wrong versions (the one from Metavision instead of the one from ECF)
    # this should be simplified once cmake supports CPACK_DEBIAN_<COMPONENT>_PACKAGE_VERSION so that we can set
    # per package/component version instead of using the same for the whole project
    foreach (comp all public)
        set(PACKAGES_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ecf/${comp}_deb/CPackConfig.cmake)
        file(COPY ${CMAKE_BINARY_DIR}/CPackConfig.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ecf/${comp}_deb)
        file(APPEND ${PACKAGES_CONFIG_FILE} "SET(CPACK_COMPONENTS_ALL hdf5-ecf-codec-lib hdf5-ecf-codec-dev hdf5-plugin-ecf hdf5-plugin-ecf-dev)\n")
        file(APPEND ${PACKAGES_CONFIG_FILE} "SET(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_BINARY_DIR}/packages/${comp})\n")
        file(APPEND ${PACKAGES_CONFIG_FILE} "SET(CPACK_DEBIAN_PACKAGE_VERSION ${HDF5_ECF_PROJECT_VERSION})\n")

        add_custom_target(${comp}_deb_packages_hdf5_ecf
            COMMAND LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_CPACK_COMMAND} -G DEB --config ${PACKAGES_CONFIG_FILE} -V
            COMMENT "Running CPack for HDF5 ECF packages"
        )
        set(${comp}_deb_packages_dependencies "${${comp}_deb_packages_dependencies};${comp}_deb_packages_hdf5_ecf" CACHE INTERNAL "" FORCE)
    endforeach (comp all public)
endif(HDF5_FOUND)