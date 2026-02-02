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

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_roi.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

void set_windows_wrapper(I_ROI &i_roi, py::list &windows_list) {
    std::vector<I_ROI::Window> windows;
    const py::ssize_t n_roi = py::len(windows_list);

    for (py::ssize_t i = 0; i < n_roi; ++i) {
        windows.push_back((windows_list[i]).cast<I_ROI::Window>());
    }

    i_roi.set_windows(windows);
}

void set_lines_wrapper(I_ROI &i_roi, py::list &cols, py::list &rows) {
    std::vector<bool> cols_vec;
    std::vector<bool> rows_vec;
    const py::ssize_t n_cols = py::len(cols);
    const py::ssize_t n_rows = py::len(rows);

    for (py::ssize_t idx = 0; idx < n_cols; ++idx) {
        cols_vec.push_back((cols[idx]).cast<bool>());
    }

    for (py::ssize_t idx = 0; idx < n_rows; ++idx) {
        rows_vec.push_back((rows[idx]).cast<bool>());
    }

    i_roi.set_lines(cols_vec, rows_vec);
}
} /* anonymous namespace */

static DeviceFacilityGetter<I_ROI> getter("get_i_roi");

static HALFacilityPythonBinder<I_ROI> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def("enable", &I_ROI::enable, py::arg("enable"), pybind_doc_hal["Metavision::I_ROI::enable"])
            .def("set_mode", &I_ROI::set_mode, py::arg("mode"), pybind_doc_hal["Metavision::I_ROI::set_mode"])
            .def("set_window", &I_ROI::set_window, py::arg("roi"), pybind_doc_hal["Metavision::I_ROI::set_window"])
            .def("get_max_supported_windows_count", &I_ROI::get_max_supported_windows_count,
                 pybind_doc_hal["Metavision::I_ROI::get_max_supported_windows_count"])
            .def("set_windows", &set_windows_wrapper, py::arg("roi_list"))
            .def("set_lines", &set_lines_wrapper, py::arg("cols"), py::arg("rows"));

        py::enum_<I_ROI::Mode>(class_binding, "Mode", py::module_local())
            .value("ROI", I_ROI::Mode::ROI)
            .value("RONI", I_ROI::Mode::RONI);

        py::class_<I_ROI::Window>(class_binding, "Window", pybind_doc_hal["Metavision::I_ROI::Window"])
            .def(py::init<const I_ROI::Window &>())
            .def(py::init<int, int, int, int>(), pybind_doc_hal["Metavision::I_ROI::Window::Window"])
            .def("to_string", &I_ROI::Window::to_string, pybind_doc_hal["Metavision::I_ROI::Window::to_string"])
            .def_readwrite("x", &I_ROI::Window::x)
            .def_readwrite("y", &I_ROI::Window::y)
            .def_readwrite("width", &I_ROI::Window::width)
            .def_readwrite("height", &I_ROI::Window::height);
    },
    "I_ROI", pybind_doc_hal["Metavision::I_ROI"]);

} // namespace Metavision
