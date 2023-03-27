/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_UI_BASE_GLFW_WINDOW_H
#define METAVISION_SDK_UI_BASE_GLFW_WINDOW_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace Metavision {

/// @brief Base class defining a UI based on GLFW and OpenGL.
///
/// This class cannot directly be used as is, instead, one needs to instantiate one of its derived class.
///
/// @warning The constructor and destructor of this class must only be called from the main thread
class BaseGLFWWindow {
public:
    /// @brief Destructor
    /// @warning Must only be called from the main thread
    virtual ~BaseGLFWWindow();

    /// @brief Indicates whether the window has been asked to close
    /// @return True if the window should close, False otherwise
    /// @note This returns the window's close flag
    bool should_close() const;

    /// @brief Asks the window to close
    /// @note This only sets the window's close flag to True but doesn't actually close the window. The window will
    /// effectively be closed when the destructor is called.
    void set_close_flag();

protected:
    /// @brief Constructs a new window
    /// @param title The window's title
    /// @param width Width of the window at starting time (can be resized later on) and width of the images that will be
    /// displayed
    /// @param height Height of the window at starting time (can be resized later on) and height of the images that will
    /// be displayed
    /// @warning Must only be called from the main thread
    BaseGLFWWindow(const std::string &title, int width, int height);

    /// @brief No copy allowed
    BaseGLFWWindow(const BaseGLFWWindow &) = delete;

    const char *glsl_version_;
    GLFWwindow *glfw_window_;
};

} // namespace Metavision

#endif // METAVISION_SDK_UI_BASE_GLFW_WINDOW_H
