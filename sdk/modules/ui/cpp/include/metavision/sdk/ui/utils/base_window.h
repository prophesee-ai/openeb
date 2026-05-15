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

#ifndef METAVISION_SDK_UI_BASE_WINDOW_H
#define METAVISION_SDK_UI_BASE_WINDOW_H

#include <functional>
#include <queue>
#include <mutex>

#include "metavision/sdk/ui/utils/base_glfw_window.h"
#include "metavision/sdk/ui/utils/ui_event.h"

namespace Metavision {

/// @brief Base class for displaying images and handling events in a window, deriving from BaseGLFWUI.
///
/// This class cannot directly be used as is, instead, one needs to instantiate one of its derived class (i.e. @ref
/// Window or @ref MTWindow).
///
/// @note Each window has its own events queue that needs to be regularly processed, either explicitly by calling
/// @ref BaseWindow::poll_events or implicitly by calling @ref Window::show or @ref MTWindow::show_async with
/// @p auto_poll = true (default behavior). Events are polled from the system and push to the windows' internal queue by
/// calling @ref EventLoop::poll_and_dispatch.
/// @warning The constructor and destructor of this class must only be called from the main thread
class BaseWindow : public BaseGLFWWindow {
public:
    using KeyCallback       = std::function<void(UIKeyEvent key, int scancode, UIAction action, int mods)>;
    using MouseCallback     = std::function<void(UIMouseButton button, UIAction action, int mods)>;
    using CursorPosCallback = std::function<void(double xpos, double ypos)>;

    /// @brief Color Rendering mode
    enum class RenderMode { GRAY, BGR };

    /// @brief Destructor
    /// @warning Must only be called from the main thread
    virtual ~BaseWindow();

    /// @brief Gets the window's current size
    /// @param width The window's width
    /// @param height The window's height
    void get_size(int &width, int &height) const;

    /// @brief Gets the window's color rendering mode
    /// @return The window's color rendering mode (Either @ref RenderMode::GRAY or @ref RenderMode::BGR)
    RenderMode get_rendering_mode() const;

    /// @brief Sets a callback that will be called when a key is pressed
    /// @param cb The callback to call on a key event
    /// @warning Due to a GLFW's limitation, this callback returns key codes corresponding to the standard US keyboard
    /// layout. However, keys corresponding to unicode characters are mapped internally to match the current keyboard
    /// layout.
    /// @note See GLFW's documentation (GLFWkeyfun) for more information
    void set_keyboard_callback(const KeyCallback &cb);

    /// @brief Sets a callback that will be called when a mouse's button is pressed
    /// @param cb The callback to call on a mouse event
    /// @note See GLFW's documentation (GLFWmousebuttonfun) for more information
    void set_mouse_callback(const MouseCallback &cb);

    /// @brief Sets a callback that will be called when the mouse's cursor moves on the current window
    /// @param cb The callback to call on a mouse cursor event
    /// @note See GLFW's documentation (GLFWcursorposfun) for more information
    void set_cursor_pos_callback(const CursorPosCallback &cb);

    /// @brief Dequeues events in this window's queue and calls corresponding callbacks
    /// @note Calling this method is not mandatory when @ref Window::show and @ref MTWindow::show_async are called with
    /// @p auto_poll = true (i.e. default behavior). However, when the image to display is somehow generated from input
    /// events (e.g. drawing ROIs from mouse inputs), then it might be useful to call this method before both generating
    /// the image and calling @ref Window::show or @ref MTWindow::show_async (with @p auto_poll = false) to reduce the
    /// latency between the moment when the events are generated and the moment when the image is actually displayed.
    /// @warning The callbacks are called in the same thread as this method's calling one. This means that if this
    /// method is called from a different thread than the one calling @ref Window::show or MTWindow::show_async, a
    /// special care must be taken, in the callbacks, to avoid concurrency problems.
    void poll_events();

protected:
    /// @brief Constructs a new window
    /// @param title The window's title
    /// @param width Width of the window at starting time (can be resized later on) and width of the images that will be
    /// displayed
    /// @param height Height of the window at starting time (can be resized later on) and height of the images that will
    /// be displayed
    /// @param mode The color rendering mode (i.e. either GRAY or BGR). Cannot be modified.
    /// @warning Must only be called from the main thread
    BaseWindow(const std::string &title, int width, int height, RenderMode mode);

    /// @brief No copy allowed
    BaseWindow(const BaseWindow &) = delete;

    /// @brief Displays the image as a textured quad
    void draw_background_texture();

    int width_;
    int height_;

    RenderMode render_mode_;

    GLuint vertex_array_id_{};
    GLuint vertex_buffer_{};
    GLuint program_id_;
    GLuint tex_id_;

private:
    struct SystemEvent {
        enum class EventType { MOUSE, KEYBOARD, CURSOR };
        EventType event_type_;

        union {
            struct {
                int mouse_btn_;
                int mouse_action_;
                int mouse_mods_;
            };
            struct {
                int keyboard_key_;
                int keyboard_scancode_;
                int keyboard_action_;
                int keyboard_mods_;
            };
            struct {
                double cursor_x_;
                double cursor_y_;
            };
        };
    };

    static void native_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void native_mouse_callback(GLFWwindow *window, int button, int action, int mods);
    static void native_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
    static void native_resize_callback(GLFWwindow *window, int width, int height);

    std::queue<SystemEvent> events_queue_back_;
    std::queue<SystemEvent> events_queue_front_;
    std::mutex events_queue_mtx_;

    KeyCallback on_key_cb_;
    MouseCallback on_mouse_cb_;
    CursorPosCallback on_cursor_pos_cb_;
};

} // namespace Metavision

#endif // METAVISION_SDK_UI_BASE_WINDOW_H
