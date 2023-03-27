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

#include <memory>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ui/utils/base_glfw_window.h"

namespace Metavision {
namespace detail {

struct GLFWContext {
    GLFWContext() {
        glfwSetErrorCallback(GLFWContext::error_callback);

        if (!glfwInit())
            throw std::runtime_error("Impossible to initialize glfw.");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        auto window = glfwCreateWindow(10, 10, "init", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("Impossible to create a glfw window");

        glfwMakeContextCurrent(window);

        if (glewInit() != GLEW_OK)
            throw std::runtime_error("Impossible to initialize GL extensions");

        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(window);
    }

    ~GLFWContext() {
        glfwTerminate();
    }

    static void error_callback(int error_code, const char *description) {
        MV_SDK_LOG_ERROR() << description;
    }
};

static bool is_glfw_initialized_ = false;
static std::unique_ptr<GLFWContext> global_glfw_context_;

} // namespace detail

BaseGLFWWindow::BaseGLFWWindow(const std::string &title, int width, int height) {
    glfw_window_ = nullptr;

    if (!detail::is_glfw_initialized_) {
        detail::global_glfw_context_ = std::make_unique<detail::GLFWContext>();
        detail::is_glfw_initialized_ = true;
    }

#ifdef __APPLE__
    glsl_version_ = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
    glsl_version_ = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    glfw_window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    glfwDefaultWindowHints();

    if (!glfw_window_)
        throw std::runtime_error("Impossible to create a glfw window");

    glfwMakeContextCurrent(glfw_window_);
}

BaseGLFWWindow::~BaseGLFWWindow() {
    if (glfw_window_) {
        glfwMakeContextCurrent(glfw_window_);
        glfwDestroyWindow(glfw_window_);
    }
}

bool BaseGLFWWindow::should_close() const {
    if (glfw_window_)
        return glfwWindowShouldClose(glfw_window_);

    return true;
}

void BaseGLFWWindow::set_close_flag() {
    if (glfw_window_)
        glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
}

} // namespace Metavision