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
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "metavision/sdk/base/utils/generic_header.h"

#include "pb_doc_base.h"

namespace py = pybind11;

namespace Metavision {

void export_generic_header(py::module &m) {
    py::class_<GenericHeader>(m, "GenericHeader")
        .def(py::init<>())
        .def(py::init([](const py::dict &dict) {
                 auto header = new GenericHeader;
                 for (const auto &pair : dict) {
                     try {
                         header->set_field(pair.first.cast<std::string>(), pair.second.cast<std::string>());
                     } catch (const py::cast_error &) {
                         std::cerr << "Error while building a RawFileHeader from a dictionary: the input dictionary "
                                      "must contain only string type for both keys and values. Failed to add field "
                                      "to header."
                                   << std::endl;
                     }
                 }
                 return header;
             }),
             "Args:\n dict (dictionary): a python dictionary holding key value pairs of string types.\n")
        .def(py::init([](const std::string filename) {
                 std::ifstream stream(filename, std::ios::binary);
                 return new GenericHeader(stream);
             }),
             "Args:\n filename (str): name of the file to open")
        .def("set_field", &GenericHeader::set_field, py::arg("key"), py::arg("value"),
             pybind_doc_base["Metavision::GenericHeader::set_field"])
        .def("get_field", &GenericHeader::get_field, py::arg("key"),
             pybind_doc_base["Metavision::GenericHeader::get_field"])
        .def("remove_field", &GenericHeader::remove_field, py::arg("key"),
             pybind_doc_base["Metavision::GenericHeader::remove_field"])
        .def("add_date", &GenericHeader::add_date, pybind_doc_base["Metavision::GenericHeader::add_date"])
        .def("get_date", &GenericHeader::get_date, pybind_doc_base["Metavision::GenericHeader::get_date"])
        .def("remove_date", &GenericHeader::remove_date, pybind_doc_base["Metavision::GenericHeader::remove_date"])
        .def("empty", &GenericHeader::empty, pybind_doc_base["Metavision::GenericHeader::empty"])
        .def("get_header_map", &GenericHeader::get_header_map,
             pybind_doc_base["Metavision::GenericHeader::get_header_map"])
        .def("to_string", &GenericHeader::to_string, pybind_doc_base["Metavision::GenericHeader::to_string"]);
}
} // namespace Metavision
