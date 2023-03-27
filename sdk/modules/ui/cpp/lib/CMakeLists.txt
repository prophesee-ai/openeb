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
        OpenGL::GL
        GLEW::GLEW)
