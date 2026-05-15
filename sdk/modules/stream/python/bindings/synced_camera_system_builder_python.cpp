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
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/synced_camera_system_builder.h"

#include "pb_doc_stream.h"

namespace py = pybind11;

namespace Metavision {

void export_synced_cameras_system_builder(py::module &m) {
    py::class_<SyncedCameraSystemBuilder>(m, "SyncedCameraSystemBuilder",
                                          pybind_doc_stream["Metavision::SyncedCameraSystemBuilder"])
        .def(py::init<>())
        .def(
            "add_live_camera_parameters",
            [](SyncedCameraSystemBuilder &builder, const std::string &serial_number, const DeviceConfig &device_config,
               const std::optional<std::filesystem::path> &settings_file_path) {
                SyncedCameraSystemFactory::LiveCameraParameters parameters;
                parameters.serial_number      = serial_number;
                parameters.device_config      = device_config;
                parameters.settings_file_path = settings_file_path;

                builder.add_live_camera_parameters(parameters);
            },
            pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::add_live_camera_parameters"],
            py::arg("serial_number"), py::arg("device_config") = DeviceConfig(),
            py::arg("settings_file_path") = std::nullopt)
        .def("set_record", &SyncedCameraSystemBuilder::set_record,
             pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::set_record"], py::arg("record"))
        .def("set_record_dir", &SyncedCameraSystemBuilder::set_record_dir,
             pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::set_record_dir"], py::arg("record_dir"))
        .def("add_record_path", &SyncedCameraSystemBuilder::add_record_path,
             pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::add_record_path"], py::arg("record_path"))
        .def("set_file_config_hints", &SyncedCameraSystemBuilder::set_file_config_hints,
             pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::set_file_config_hints"],
             py::arg("file_config_hints"))
        .def("build", &SyncedCameraSystemBuilder::build,
             pybind_doc_stream["Metavision::SyncedCameraSystemBuilder::build"]);
}
} // namespace Metavision
