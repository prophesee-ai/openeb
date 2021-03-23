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
#include <pybind11/operators.h>

#include "hal_python_binder.h"
#include "metavision/hal/utils/device_roi.h"
#include "pb_doc_hal.h"

namespace py = pybind11;

namespace Metavision {

static HALClassPythonBinder<DeviceRoi> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def(py::init<const DeviceRoi &>())
            .def(py::init<int, int, int, int>(), pybind_doc_hal["Metavision::DeviceRoi::DeviceRoi"])
            .def(py::self == py::self)
            .def("to_string", &DeviceRoi::to_string, pybind_doc_hal["Metavision::DeviceRoi::to_string"])
            .def_readwrite("x", &DeviceRoi::x_)
            .def_readwrite("y", &DeviceRoi::y_)
            .def_readwrite("width", &DeviceRoi::width_)
            .def_readwrite("height", &DeviceRoi::height_);
    },
    "DeviceRoi", pybind_doc_hal["Metavision::DeviceRoi"]);

} // namespace Metavision
