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

#include <pybind11/pybind11.h>

#include "metavision/sdk/ui/utils/base_glfw_window.h"
#include "metavision/sdk/ui/utils/base_window.h"

#include "pb_doc_ui.h"
#include "window_wrapper.h"

namespace py = pybind11;

namespace Metavision {

void export_base_window(py::module &m) {
    py::class_<BaseWindowWrapper> base_window(m, "BaseWindow", pybind_doc_ui["Metavision::BaseWindow"]);

    base_window
        .def(
            "get_size",
            [](const BaseWindowWrapper &base_window) {
                int width, height;
                base_window.get<BaseWindow>()->get_size(width, height);
                return py::make_tuple(width, height);
            },
            "Returns the tuple (widh, height)")
        .def(
            "get_rendering_mode",
            [](const BaseWindowWrapper &base_window) { return base_window.get<BaseWindow>()->get_rendering_mode(); },
            pybind_doc_ui["Metavision::BaseWindow::get_rendering_mode"])
        .def(
            "should_close",
            [](const BaseWindowWrapper &base_window) { return base_window.get<BaseWindow>()->should_close(); },
            pybind_doc_ui["Metavision::BaseGLFWWindow::should_close"])
        .def(
            "set_close_flag", [](BaseWindowWrapper &base_window) { base_window.get<BaseWindow>()->set_close_flag(); },
            pybind_doc_ui["Metavision::BaseGLFWWindow::set_close_flag"])
        .def(
            "set_keyboard_callback",
            [](BaseWindowWrapper &base_window, const py::object &object) {
                BaseWindow::KeyCallback cb = [object](int key, int scancode, int action, int mods) {
                    object(key, scancode, action, mods);
                };
                base_window.get<BaseWindow>()->set_keyboard_callback(cb);
            },
            pybind_doc_ui["Metavision::BaseWindow::set_keyboard_callback"])
        .def(
            "set_mouse_callback",
            [](BaseWindowWrapper &base_window, const py::object &object) {
                BaseWindow::MouseCallback cb = [object](int button, int action, int mods) {
                    object(button, action, mods);
                };
                base_window.get<BaseWindow>()->set_mouse_callback(cb);
            },
            pybind_doc_ui["Metavision::BaseWindow::set_mouse_callback"])
        .def(
            "set_cursor_pos_callback",
            [](BaseWindowWrapper &base_window, const py::object &object) {
                BaseWindow::CursorPosCallback cb = [object](double xpos, double ypos) { object(xpos, ypos); };
                base_window.get<BaseWindow>()->set_cursor_pos_callback(cb);
            },
            pybind_doc_ui["Metavision::BaseWindow::set_cursor_pos_callback"])
        .def(
            "poll_events", [](BaseWindowWrapper &base_window) { base_window.get<BaseWindow>()->poll_events(); },
            pybind_doc_ui["Metavision::BaseWindow::poll_events"]);

    py::enum_<BaseWindow::RenderMode>(base_window, "RenderMode")
        .value("GRAY", BaseWindow::RenderMode::GRAY)
        .value("BGR", BaseWindow::RenderMode::BGR)
        .export_values();
}

} // namespace Metavision
