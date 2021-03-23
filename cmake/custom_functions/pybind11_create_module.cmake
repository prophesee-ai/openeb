# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

function(pybind11_create_module module module_output_name)
    set(module_name metavision_${module})
    string(REPLACE "_" "-" module-name ${module_name})
    set(module-name-python "${module-name}")
    if(module_name MATCHES "(.*sdk*|.*hal*)")
        set(module-name-python "${module-name}-python")
    endif()
    if(WIN32)
	    file(COPY __init__.py
	         DESTINATION ${CMAKE_BINARY_DIR}/py3/${CMAKE_BUILD_TYPE}/${module_name}
	         FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
	    set(module_name metavision_${module}_internal)
	    string(REPLACE "_" "-" module-name ${module_name})
	    set(module-name-python "${module-name}")
	    if(module_name MATCHES "(.*sdk*|.*hal*)")
   	        set(module-name-python "${module-name}-python")
   	    endif()
	    pybind11_install(DIRECTORY ${CMAKE_BINARY_DIR}/py3/${CMAKE_BUILD_TYPE}/metavision_${module}
	    	DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
	    	COMPONENT ${module-name-python}-local-install)
    endif()

    set(${module_output_name} ${module_name} PARENT_SCOPE)
    pybind11_add_module(${module_name}_python3 MODULE)

    pybind11_set_target_properties(${module_name}_python3
        PROPERTIES
            OUTPUT_NAME "${module_name}"
            LIBRARY_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${PYTHON3_OUTPUT_DIR}
    )
     
    pybind11_install(TARGETS ${module_name}_python3
            RUNTIME
                DESTINATION "PYTHON_SYSTEM_SITE_PACKAGES"
                COMPONENT ${module-name-python}
                EXCLUDE_FROM_ALL
            LIBRARY
                DESTINATION "PYTHON_SYSTEM_SITE_PACKAGES"
                COMPONENT ${module-name-python}
                EXCLUDE_FROM_ALL
            ARCHIVE
                DESTINATION "PYTHON_SYSTEM_SITE_PACKAGES"
                COMPONENT ${module-name-python}
                EXCLUDE_FROM_ALL
    )

    pybind11_install(TARGETS ${module_name}_python3
            RUNTIME
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT ${module-name-python}-local-install
            LIBRARY
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT ${module-name-python}-local-install
            ARCHIVE
                DESTINATION "PYTHON_LOCAL_SITE_PACKAGES"
                COMPONENT ${module-name-python}-local-install
    )
endfunction()
