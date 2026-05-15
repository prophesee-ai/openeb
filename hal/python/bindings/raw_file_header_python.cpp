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

#include <iostream>
#include "hal_python_binder.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "pb_doc_hal.h"

namespace Metavision {

static HALClassPythonBinder<RawFileHeader, PythonBases<GenericHeader>> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def(py::init<>())
            .def(py::init<const RawFileHeader &>())
            .def(
                py::init([](const py::dict &dict) {
                    auto header = new RawFileHeader;
                    for (const auto &pair : dict) {
                        try {
                            header->set_field(pair.first.cast<std::string>(), pair.second.cast<std::string>());
                        } catch (const py::cast_error &e) {
                            std::cerr << "Error while building a RawFileHeader from a dictionary: the input dictionary "
                                         "must contain only string type for both keys and values. Failed to add field "
                                         "to header."
                                      << std::endl;
                        }
                    }
                    return header;
                }),
                "Args:\n"
                "    dict (dictionary): a python dictionary holding key value pairs of string types.\n")
            .def("set_camera_integrator_name", &RawFileHeader::set_camera_integrator_name, py::arg("integrator_name"),
                 pybind_doc_hal["Metavision::RawFileHeader::set_camera_integrator_name"])
            .def("get_camera_integrator_name", &RawFileHeader::get_camera_integrator_name,
                 pybind_doc_hal["Metavision::RawFileHeader::get_camera_integrator_name"])
            .def("set_plugin_integrator_name", &RawFileHeader::set_plugin_integrator_name, py::arg("integrator_name"),
                 pybind_doc_hal["Metavision::RawFileHeader::set_plugin_integrator_name"])
            .def("get_plugin_integrator_name", &RawFileHeader::get_plugin_integrator_name,
                 pybind_doc_hal["Metavision::RawFileHeader::get_plugin_integrator_name"])
            .def("set_plugin_name", &RawFileHeader::set_plugin_name, py::arg("plugin_name"),
                 pybind_doc_hal["Metavision::RawFileHeader::set_plugin_name"])
            .def("get_plugin_name", &RawFileHeader::get_plugin_name,
                 pybind_doc_hal["Metavision::RawFileHeader::get_plugin_name"]);
    },
    "RawFileHeader", pybind_doc_hal["Metavision::RawFileHeader"]);

} // namespace Metavision
