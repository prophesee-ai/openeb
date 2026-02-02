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
#include "metavision/sdk/stream/synced_camera_system_factory.h"

#include "pb_doc_stream.h"

namespace py = pybind11;

namespace Metavision {

void export_synced_cameras_system_factory(py::module &m) {
    auto outer_class =
        py::class_<SyncedCameraSystemFactory>(m, "SyncedCameraSystemFactory",
                                              pybind_doc_stream["Metavision::SyncedCameraSystemFactory"])
            .def(py::init<>())
            .def_static(
                "build",
                py::overload_cast<const SyncedCameraSystemFactory::LiveParameters &>(&SyncedCameraSystemFactory::build),
                pybind_doc_stream["Metavision::SyncedCameraSystemFactory::build(const LiveParameters &parameters)"],
                py::arg("parameters"))

            .def_static(
                "build",
                py::overload_cast<const SyncedCameraSystemFactory::OfflineParameters &>(
                    &SyncedCameraSystemFactory::build),
                pybind_doc_stream["Metavision::SyncedCameraSystemFactory::build(const OfflineParameters &parameters)"],
                py::arg("parameters"));

    py::class_<SyncedCameraSystemFactory::LiveCameraParameters>(
        outer_class, "LiveCameraParameters",
        pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveCameraParameters"])
        .def(py::init<>())
        .def_readwrite("serial_number", &SyncedCameraSystemFactory::LiveCameraParameters::serial_number,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveCameraParameters::serial_number"])
        .def_readwrite("device_config", &SyncedCameraSystemFactory::LiveCameraParameters::device_config,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveCameraParameters::device_config"])
        .def_readwrite(
            "settings_file_path", &SyncedCameraSystemFactory::LiveCameraParameters::settings_file_path,
            pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveCameraParameters::settings_file_path"]);

    py::class_<SyncedCameraSystemFactory::LiveParameters>(
        outer_class, "LiveParameters", pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveParameters"])
        .def(py::init<>())
        .def_readwrite("master_parameters", &SyncedCameraSystemFactory::LiveParameters::master_parameters,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveParameters::master_parameters"])
        .def_readwrite("slave_parameters", &SyncedCameraSystemFactory::LiveParameters::slave_parameters,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveParameters::slave_parameters"])
        .def_readwrite("record", &SyncedCameraSystemFactory::LiveParameters::record,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveParameters::record"])
        .def_readwrite("record_dir", &SyncedCameraSystemFactory::LiveParameters::record_dir,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::LiveParameters::record_dir"]);

    py::class_<SyncedCameraSystemFactory::OfflineParameters>(
        outer_class, "OfflineParameters", pybind_doc_stream["Metavision::SyncedCameraSystemFactory::OfflineParameters"])
        .def(py::init<>())
        .def_readwrite("master_file_path", &SyncedCameraSystemFactory::OfflineParameters::master_file_path,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::OfflineParameters::master_file_path"])
        .def_readwrite("slave_file_paths", &SyncedCameraSystemFactory::OfflineParameters::slave_file_paths,
                       pybind_doc_stream["Metavision::SyncedCameraSystemFactory::OfflineParameters::slave_file_paths"])
        .def_readwrite(
            "file_config_hints", &SyncedCameraSystemFactory::OfflineParameters::file_config_hints,
            pybind_doc_stream["Metavision::SyncedCameraSystemFactory::OfflineParameters::file_config_hints"]);
}
} // namespace Metavision
