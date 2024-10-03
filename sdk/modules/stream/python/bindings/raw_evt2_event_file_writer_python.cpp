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

#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/raw_evt2_event_file_writer.h"
#include "pb_doc_stream.h"

namespace py = pybind11;

namespace Metavision {

void export_raw_evt2_event_file_writer(py::module &m) {
    py::class_<RAWEvt2EventFileWriter>(m, "RAWEvt2EventFileWriter",
                                       pybind_doc_stream["Metavision::RAWEvt2EventFileWriter"])
        .def(py::init<int, int, const std::filesystem::path &, bool,
                      const std::unordered_map<std::string, std::string> &, timestamp>(),
             py::arg("stream_width"), py::arg("stream_height"), py::arg("path") = std::filesystem::path(),
             py::arg("enable_trigger_support") = false,
             py::arg("metadata_map")           = std::unordered_map<std::string, std::string>(),
             py::arg("max_events_add_latency") = std::numeric_limits<timestamp>::max())
        .def("open", &RAWEvt2EventFileWriter::open, py::arg("path"),
             pybind_doc_stream["Metavision::EventFileWriter::open"])
        .def("close", &RAWEvt2EventFileWriter::close, pybind_doc_stream["Metavision::EventFileWriter::close"])
        .def("is_open", &RAWEvt2EventFileWriter::is_open, pybind_doc_stream["Metavision::EventFileWriter::is_open"])
        .def("flush", &RAWEvt2EventFileWriter::flush, pybind_doc_stream["Metavision::EventFileWriter::flush"])
        .def("add_metadata", &RAWEvt2EventFileWriter::add_metadata, py::arg("key"), py::arg("value"),
             pybind_doc_stream["Metavision::EventFileWriter::add_metadata"])
        .def(
            "add_cd_events",
            [](RAWEvt2EventFileWriter &writer, py::array_t<EventCD> buffer) {
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
            [](RAWEvt2EventFileWriter &writer, py::array_t<EventExtTrigger> buffer) {
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
