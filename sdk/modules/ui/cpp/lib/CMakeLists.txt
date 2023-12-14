# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# GLEW
find_package(GLEW REQUIRED)

# GLFW
find_package(glfw3 REQUIRED)

MetavisionSDK_add_module(ui
    REQUIRED_METAVISION_SDK_MODULES
        PUBLIC
            core
    EXTRA_REQUIRED_PACKAGE "OpenCV COMPONENTS core"
    EXTRA_REQUIRED_PACKAGE "OpenGL"
    EXTRA_REQUIRED_PACKAGE "GLEW"
    EXTRA_REQUIRED_PACKAGE "glfw3"

    VARIABLE_TO_SET "OpenGL_GL_PREFERENCE GLVND"
)

target_link_libraries(metavision_sdk_ui
    PUBLIC
        opencv_core
        glfw
)

option(USE_OPENGL_ES3 "Use OpenGL ES3 instead of OpenGl Desktop (usefull on embedded devices)" OFF)

if(USE_OPENGL_ES3)
    add_definitions(-D_USE_OPENGL_ES3_)
    if(NOT TARGET OpenGL::GLES3)
        ## Previous to cmake 3.27, OpenGL ES targets were not supported...
        ## Let's search for it
        find_library(GLES3_lib GLESv2) # GLES3 is a superset of GLES2 hence the GLESv2 library
        find_file(GLES3_header gl3.h PATH_SUFFIXES GLES3)
        get_filename_component(GLES3_header_dir ${GLES3_header} DIRECTORY)
        if(GLES_lib MATCHES "NOTFOUND")
            message(NOTICE "The required OpenGL ES3 library is not found")
            return()
        endif() 
        if(GLES3_header MATCHES "NOTFOUND")
            message(NOTICE "Could not found OpenGL ES3 headers")
            return()
        endif()
        add_library(GLES3 UNKNOWN IMPORTED GLOBAL) 
        set_target_properties(GLES3 PROPERTIES 
            IMPORTED_LOCATION ${GLES3_lib}
            INTERFACE_INCLUDE_DIRECTORIES ${GLES3_header_dir}
        )
        add_library(OpenGL::GLES3 ALIAS GLES3)
    endif()

    target_link_libraries(metavision_sdk_ui PUBLIC OpenGL::GLES3)
else()
    target_link_libraries(metavision_sdk_ui 
        PUBLIC 
            OpenGL::GL 
            GLEW::GLEW
    )
endif(USE_OPENGL_ES3)


