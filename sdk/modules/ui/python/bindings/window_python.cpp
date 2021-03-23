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
#include <pybind11/numpy.h>

#include "metavision/sdk/ui/utils/window.h"

#include "pb_doc_ui.h"
#include "window_wrapper.h"

namespace py = pybind11;

namespace Metavision {

void export_window(py::module &m) {
    using namespace pybind11::literals;

    py::class_<WindowWrapper, BaseWindowWrapper>(m, "Window", pybind_doc_ui["Metavision::Window"])
        .def(py::init(
                 [](const std::string &title, int width, int height, BaseWindow::RenderMode mode, bool open_directly) {
                     WindowWrapper *ptr = new WindowWrapper(title, width, height, mode);
                     if (open_directly)
                         ptr->enter();
                     return ptr;
                 }),
             "title"_a, "width"_a, "height"_a, "mode"_a, "open_directly"_a = false)
        .def(
            "show",
            [](WindowWrapper &window, const py::array &image, bool auto_poll) {
                if (!py::isinstance<py::array_t<std::uint8_t>>(image))
                    throw std::invalid_argument("Incompatible input dtype. Must be np.ubyte.");

                Window *window_ptr = window.get<Window>();
                if (!window_ptr)
                    throw std::logic_error(
                        "The window must be open to call the show method. Use open_directly=True "
                        "when constructing the window or instantiate it using the 'with' statement.");

                int mat_type;
                if (window_ptr->get_rendering_mode() == BaseWindow::RenderMode::GRAY) {
                    mat_type = CV_8UC1;
                    if (image.ndim() != 2)
                        throw std::invalid_argument(
                            "Incompatible input map's dimensions number. Must be a 2 dimensional map.");
                } else {
                    mat_type = CV_8UC3;
                    if (image.ndim() != 3)
                        throw std::invalid_argument(
                            "Incompatible input map's dimensions number. Must be a 3 dimensional map.");
                }

                const auto &shape   = image.shape();
                const auto &strides = image.strides();
                auto *ptr           = image.request().ptr;

                const cv::Mat image_cv(shape[0], shape[1], mat_type, ptr, strides[0]);

                window_ptr->show(image_cv, auto_poll);
            },
            "image"_a, "auto_poll"_a = true, pybind_doc_ui["Metavision::Window::show"])
        .def(
            "__enter__",
            [](WindowWrapper &window, py::args) {
                window.enter();
                return &window;
            },
            "Method that is invoked on entry to the body of the 'with' statement")
        .def(
            "__exit__", [](WindowWrapper &window, py::args) { window.exit(); },
            "Method that is invoked on exit from the body of the 'with' statement")
        .def(
            "destroy", [](WindowWrapper &window) { window.exit(); },
            "Destroys the window. This method has to be called after closing the window "
            "if this class has been constructed using open_directly=True.");
}

} // namespace Metavision