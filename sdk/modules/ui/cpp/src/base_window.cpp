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

#include <map>
#include <memory>

#include "metavision/sdk/ui/utils/opengl_api.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ui/detail/texture_utils.h"
#include "metavision/sdk/ui/utils/base_window.h"

static const std::map<char, int> name_to_key = {
    {'a', GLFW_KEY_A}, {'z', GLFW_KEY_Z}, {'e', GLFW_KEY_E},    {'r', GLFW_KEY_R}, {'t', GLFW_KEY_T}, {'y', GLFW_KEY_Y},
    {'u', GLFW_KEY_U}, {'i', GLFW_KEY_I}, {'o', GLFW_KEY_O},    {'p', GLFW_KEY_P}, {'m', GLFW_KEY_M}, {'l', GLFW_KEY_L},
    {'k', GLFW_KEY_K}, {'j', GLFW_KEY_J}, {'h', GLFW_KEY_H},    {'g', GLFW_KEY_G}, {'f', GLFW_KEY_F}, {'d', GLFW_KEY_D},
    {'s', GLFW_KEY_S}, {'q', GLFW_KEY_Q}, {'w', GLFW_KEY_W},    {'x', GLFW_KEY_X}, {'c', GLFW_KEY_C}, {'v', GLFW_KEY_V},
    {'b', GLFW_KEY_B}, {'n', GLFW_KEY_N}, {',', GLFW_KEY_COMMA}};

namespace Metavision {
namespace detail {

GLuint LoadShaders() {
    const char *vertex_shader_str = "#version 310 es\n"
                                    "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                                    "layout(location = 1) in vec2 vertexUV;\n"
                                    "out vec2 UV;\n"
                                    "void main(){\n"
                                    "    gl_Position.xyz = vertexPosition_modelspace;\n"
                                    "    gl_Position.w = 1.0;\n"
                                    "    UV = vertexUV;\n"
                                    "}\n";

    const char *fragment_shader_str = "#version 310 es\n"
                                      "in mediump vec2 UV;\n"
                                      "out mediump vec3 color;\n"
                                      "uniform sampler2D Sampler;\n"
                                      "void main(){\n"
                                      "     // Inputs are filled using BGR format\n"
                                      "    color = texture( Sampler, UV ).bgr;\n"
                                      "}\n";

    // Create the shaders
    GLuint vertex_shader_id   = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

    GLint result = GL_FALSE;
    int info_log_length;

    // Compile Vertex Shader
    glShaderSource(vertex_shader_id, 1, &vertex_shader_str, NULL);
    glCompileShader(vertex_shader_id);

    // Check Vertex Shader
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertex_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::string VertexShaderErrorMessage;
        VertexShaderErrorMessage.resize(info_log_length + 1);
        glGetShaderInfoLog(vertex_shader_id, info_log_length, NULL, &VertexShaderErrorMessage[0]);
        MV_SDK_LOG_ERROR() << VertexShaderErrorMessage;
    }

    // Compile Fragment Shader
    glShaderSource(fragment_shader_id, 1, &fragment_shader_str, NULL);
    glCompileShader(fragment_shader_id);

    // Check Fragment Shader
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::string FragmentShaderErrorMessage;
        FragmentShaderErrorMessage.resize(info_log_length + 1);
        glGetShaderInfoLog(fragment_shader_id, info_log_length, NULL, &FragmentShaderErrorMessage[0]);
        MV_SDK_LOG_ERROR() << FragmentShaderErrorMessage;
    }

    // Link the program
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, fragment_shader_id);
    glLinkProgram(program_id);

    // Check the program
    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::string ProgramErrorMessage;
        ProgramErrorMessage.resize(info_log_length + 1);
        glGetProgramInfoLog(program_id, info_log_length, NULL, &ProgramErrorMessage[0]);
        MV_SDK_LOG_ERROR() << ProgramErrorMessage;
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, fragment_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    return program_id;
}

} // namespace detail

BaseWindow::BaseWindow(const std::string &title, int width, int height, RenderMode mode) :
    BaseGLFWWindow(title, width, height), width_(width), height_(height), render_mode_(mode) {
    const detail::TextureOptions texture_options{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                                                 (render_mode_ == RenderMode::GRAY) ? detail::TextureFormat::Gray :
                                                                                      detail::TextureFormat::RGB,
                                                 detail::TextureFilter::Linear, detail::TextureFilter::Linear};
    program_id_ = detail::LoadShaders();
    tex_id_     = detail::initialize_texture(texture_options);

    // clang-format off
    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f
    };
    // clang-format on

    glGenVertexArrays(1, &vertex_array_id_);
    glBindVertexArray(vertex_array_id_);

    glGenBuffers(1, &vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glfwMakeContextCurrent(nullptr);

    glfwSetWindowUserPointer(glfw_window_, this);
    glfwSetFramebufferSizeCallback(glfw_window_, native_resize_callback);
    glfwSetWindowAspectRatio(glfw_window_, width, height);
    glfwSetWindowSizeLimits(glfw_window_, 200, 200 * height / width, GLFW_DONT_CARE, GLFW_DONT_CARE);
}

BaseWindow::~BaseWindow() {
    if (glfw_window_) {
        glfwMakeContextCurrent(glfw_window_);
        glDeleteBuffers(1, &vertex_buffer_);
        glDeleteVertexArrays(1, &vertex_array_id_);
        glDeleteTextures(1, &tex_id_);
        glDeleteProgram(program_id_);
    }
}

void BaseWindow::get_size(int &width, int &height) const {
    width  = width_;
    height = height_;
}

BaseWindow::RenderMode BaseWindow::get_rendering_mode() const {
    return render_mode_;
}

void BaseWindow::set_keyboard_callback(const KeyCallback &cb) {
    on_key_cb_ = cb;

    glfwSetKeyCallback(glfw_window_, native_key_callback);
}

void BaseWindow::set_mouse_callback(const MouseCallback &cb) {
    on_mouse_cb_ = cb;

    glfwSetMouseButtonCallback(glfw_window_, native_mouse_callback);
}

void BaseWindow::set_cursor_pos_callback(const CursorPosCallback &cb) {
    on_cursor_pos_cb_ = cb;

    glfwSetCursorPosCallback(glfw_window_, native_cursor_pos_callback);
}

void BaseWindow::poll_events() {
    {
        std::lock_guard<std::mutex> lock(events_queue_mtx_);
        std::swap(events_queue_front_, events_queue_back_);
    }

    while (!events_queue_back_.empty()) {
        const auto event = events_queue_back_.front();
        events_queue_back_.pop();

        switch (event.event_type_) {
        case SystemEvent::EventType::KEYBOARD:
            on_key_cb_(static_cast<UIKeyEvent>(event.keyboard_key_), event.keyboard_scancode_,
                       static_cast<UIAction>(event.keyboard_action_), event.keyboard_mods_);
            break;

        case SystemEvent::EventType::MOUSE:
            on_mouse_cb_(static_cast<UIMouseButton>(event.mouse_btn_), static_cast<UIAction>(event.mouse_action_),
                         event.mouse_mods_);
            break;

        case SystemEvent::EventType::CURSOR:
            on_cursor_pos_cb_(event.cursor_x_, event.cursor_y_);
            break;
        }
    }
}

void BaseWindow::draw_background_texture() {
    int width, height;
    glfwGetFramebufferSize(glfw_window_, &width, &height);

    glViewport(0, 0, width, height);

    glUseProgram(program_id_);

    glBindTexture(GL_TEXTURE_2D, tex_id_);

    glBindVertexArray(vertex_array_id_);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(glfw_window_);
}

void BaseWindow::native_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto *instance = reinterpret_cast<BaseWindow *>(glfwGetWindowUserPointer(window));

    int remapped_key = key;
    const char *c    = glfwGetKeyName(key, scancode);
    if (c != nullptr) {
        auto new_key = name_to_key.find(*c);
        if (new_key != name_to_key.end())
            remapped_key = new_key->second;
    }

    SystemEvent event;
    event.event_type_        = SystemEvent::EventType::KEYBOARD;
    event.keyboard_key_      = remapped_key;
    event.keyboard_scancode_ = scancode;
    event.keyboard_action_   = action;
    event.keyboard_mods_     = mods;

    std::lock_guard<std::mutex> lock(instance->events_queue_mtx_);
    instance->events_queue_front_.push(event);
}

void BaseWindow::native_mouse_callback(GLFWwindow *window, int button, int action, int mods) {
    auto *instance = reinterpret_cast<BaseWindow *>(glfwGetWindowUserPointer(window));

    SystemEvent event;
    event.event_type_   = SystemEvent::EventType::MOUSE;
    event.mouse_btn_    = button;
    event.mouse_action_ = action;
    event.mouse_mods_   = mods;

    std::lock_guard<std::mutex> lock(instance->events_queue_mtx_);
    instance->events_queue_front_.push(event);
}

void BaseWindow::native_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    auto *instance = reinterpret_cast<BaseWindow *>(glfwGetWindowUserPointer(window));

    SystemEvent event;
    event.event_type_ = SystemEvent::EventType::CURSOR;
    event.cursor_x_   = xpos;
    event.cursor_y_   = ypos;

    std::lock_guard<std::mutex> lock(instance->events_queue_mtx_);
    instance->events_queue_front_.push(event);
}

void BaseWindow::native_resize_callback(GLFWwindow *window, int width, int height) {
    auto *instance = reinterpret_cast<BaseWindow *>(glfwGetWindowUserPointer(window));

    instance->width_  = width;
    instance->height_ = height;
}

} // namespace Metavision
