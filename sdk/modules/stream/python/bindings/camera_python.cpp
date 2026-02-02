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
#include <pybind11/stl/filesystem.h>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/file_config_hints.h"
#include "pb_doc_stream.h"
#include "rvalue_camera.h"

namespace py = pybind11;

namespace Metavision {

void export_camera(py::module &m) {
    py::class_<FileConfigHints>(m, "FileConfigHints", pybind_doc_stream["Metavision::FileConfigHints"])
        .def(py::init<>(), pybind_doc_stream["Metavision::FileConfigHints::FileConfigHints"])
        .def("real_time_playback", static_cast<bool (FileConfigHints::*)() const>(&FileConfigHints::real_time_playback),
             pybind_doc_stream["Metavision::FileConfigHints::real_time_playback() const"])
        .def("real_time_playback", py::overload_cast<bool>(&FileConfigHints::real_time_playback), py::arg("enabled"),
             pybind_doc_stream["Metavision::FileConfigHints::real_time_playback(bool enabled)"])
        .def("time_shift", static_cast<bool (FileConfigHints::*)() const>(&FileConfigHints::time_shift),
             pybind_doc_stream["Metavision::FileConfigHints::time_shift() const"])
        .def("time_shift", py::overload_cast<bool>(&FileConfigHints::time_shift), py::arg("enabled"),
             pybind_doc_stream["Metavision::FileConfigHints::time_shift(bool enabled)"])
        .def("max_memory", static_cast<std::size_t (FileConfigHints::*)() const>(&FileConfigHints::max_memory),
             pybind_doc_stream["Metavision::FileConfigHints::max_memory() const"])
        .def("max_memory", py::overload_cast<std::size_t>(&FileConfigHints::max_memory), py::arg("max_memory"),
             pybind_doc_stream["Metavision::FileConfigHints::max_memory(std::size_t max_memory)"])
        .def("max_read_per_op",
             static_cast<std::size_t (FileConfigHints::*)() const>(&FileConfigHints::max_read_per_op),
             pybind_doc_stream["Metavision::FileConfigHints::max_read_per_op() const"])
        .def("max_memory", py::overload_cast<std::size_t>(&FileConfigHints::max_read_per_op),
             py::arg("max_read_per_op"),
             pybind_doc_stream["Metavision::FileConfigHints::max_read_per_op(std::size_t max_read_per_op)"])
        .def("set",
             static_cast<void (FileConfigHints::*)(const std::string &, const std::string &)>(&FileConfigHints::set),
             py::arg("key"), py::arg("value"),
             pybind_doc_stream["Metavision::FileConfigHints::set(const std::string &key, const std::string &value)"])
        .def("get", &FileConfigHints::get<std::string>, py::arg("key"), py::arg("def") = std::string(),
             pybind_doc_stream["Metavision::FileConfigHints::get"]);

    py::class_<RValueCamera>(m, "RValueCamera");

    py::class_<Camera>(m, "Camera", pybind_doc_stream["Metavision::Camera"])
        .def_static("from_first_available", py::overload_cast<>(&Camera::from_first_available),
                    pybind_doc_stream["Metavision::Camera::from_first_available()"])
        .def_static("from_first_available", py::overload_cast<const DeviceConfig &>(&Camera::from_first_available),
                    py::arg("config"),
                    pybind_doc_stream["Metavision::Camera::from_first_available(const DeviceConfig &config)"])
        .def_static("from_serial", py::overload_cast<const std::string &>(&Camera::from_serial), py::arg("serial"),
                    pybind_doc_stream["Metavision::Camera::from_serial(const std::string &serial)"])
        .def_static(
            "from_serial", py::overload_cast<const std::string &, const DeviceConfig &>(&Camera::from_serial),
            py::arg("serial"), py::arg("config"),
            pybind_doc_stream["Metavision::Camera::from_serial(const std::string &serial, const DeviceConfig &config)"])
        .def_static("from_file", &Camera::from_file, py::arg("file_path"), py::arg("hints") = FileConfigHints(),
                    pybind_doc_stream["Metavision::Camera::from_file"])
        .def("get_device", static_cast<Device &(Camera::*)()>(&Camera::get_device), py::return_value_policy::reference,
             pybind_doc_stream["Metavision::Camera::get_device"])
        .def("save", &Camera::save, py::arg("path"), pybind_doc_stream["Metavision::Camera::save"])
        .def("load", &Camera::load, py::arg("path"), pybind_doc_stream["Metavision::Camera::load"])
        .def(
            "width", [](const Camera &camera) { return camera.geometry().get_width(); },
            "Returns the width of the camera\n")
        .def(
            "height", [](const Camera &camera) { return camera.geometry().get_height(); },
            "Returns the height of the camera\n")
        .def("move", [](Camera &camera) { return RValueCamera{std::move(camera)}; });
}

} // namespace Metavision
