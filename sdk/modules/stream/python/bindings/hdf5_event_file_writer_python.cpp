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
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/hdf5_event_file_writer.h"
#include "pb_doc_stream.h"

namespace py = pybind11;

namespace Metavision {

void export_hdf5_event_file_writer(py::module &m) {
    py::class_<HDF5EventFileWriter>(m, "HDF5EventFileWriter", pybind_doc_stream["Metavision::HDF5EventFileWriter"])
        .def(py::init<const std::filesystem::path &, const std::unordered_map<std::string, std::string>>(),
             py::arg("path") = "", py::arg("metadata_map") = std::unordered_map<std::string, std::string>())
        .def("open", &HDF5EventFileWriter::open, py::arg("path"),
             pybind_doc_stream["Metavision::EventFileWriter::open"])
        .def("close", &HDF5EventFileWriter::close, pybind_doc_stream["Metavision::EventFileWriter::close"])
        .def("is_open", &HDF5EventFileWriter::is_open, pybind_doc_stream["Metavision::EventFileWriter::is_open"])
        .def("flush", &HDF5EventFileWriter::flush, pybind_doc_stream["Metavision::EventFileWriter::flush"])
        .def("add_metadata", &HDF5EventFileWriter::add_metadata, py::arg("key"), py::arg("value"),
             pybind_doc_stream["Metavision::EventFileWriter::add_metadata"])
        .def("add_metadata_map_from_camera", &HDF5EventFileWriter::add_metadata_map_from_camera, py::arg("camera"),
             pybind_doc_stream["Metavision::EventFileWriter::add_metadata_map_from_camera"])
        .def(
            "add_cd_events",
            [](HDF5EventFileWriter &writer, py::array_t<EventCD> buffer) {
                py::buffer_info buffer_info = buffer.request();
                if (buffer_info.ndim > 1) {
                    throw std::invalid_argument("EventFileWriter.add_cd_events expects one dimentional arrays");
                }
                if (buffer_info.itemsize != sizeof(EventCD)) {
                    throw std::invalid_argument("EventFileWriter.add_cd_events received array with invalid items");
                }
                bool res = writer.add_events(static_cast<const EventCD *>(buffer_info.ptr),
                                             static_cast<const EventCD *>(buffer_info.ptr) + buffer_info.size);
            },
            py ::arg("events"),
            "Adds an array of EventCD to write to the file\n"
            "\n"
            "Args:\n"
            "    events: numpy array of EventCD\n")
        .def(
            "add_ext_trigger_events",
            [](HDF5EventFileWriter &writer, py::array_t<EventExtTrigger> buffer) {
                py::buffer_info buffer_info = buffer.request();
                if (buffer_info.ndim > 1) {
                    throw std::invalid_argument(
                        "EventFileWriter.add_ext_trigger_events expects one dimentional arrays");
                }
                if (buffer_info.itemsize != sizeof(EventExtTrigger)) {
                    throw std::invalid_argument(
                        "EventFileWriter.add_ext_trigger_events received array with invalid items");
                }
                bool res = writer.add_events(static_cast<const EventExtTrigger *>(buffer_info.ptr),
                                             static_cast<const EventExtTrigger *>(buffer_info.ptr) + buffer_info.size);
            },
            py ::arg("events"),
            "Adds an array of EventExtTrigger to write to the file\n"
            "\n"
            "Args:\n"
            "    events: numpy array of EventExtTrigger\n");
}

} // namespace Metavision
