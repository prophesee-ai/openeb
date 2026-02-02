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

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "hal_python_binder.h"
#include "metavision/hal/device/device_discovery.h"
#include "pb_doc_hal.h"

namespace py = pybind11;

namespace Metavision {

namespace {

std::shared_ptr<Device> open_with_only_serial(const std::string &serial) {
    return std::shared_ptr<Device>(DeviceDiscovery::open(serial));
}

std::shared_ptr<Device> open_with_config_wrapper(const std::string &serial, DeviceConfig &config) {
    return std::shared_ptr<Device>(DeviceDiscovery::open(serial, config));
}

std::shared_ptr<Device> open_raw_file_with_only_filename(const std::filesystem::path &raw_file) {
    return std::shared_ptr<Device>(DeviceDiscovery::open_raw_file(raw_file));
}

std::shared_ptr<Device> open_raw_file_with_config_wrapper(const std::filesystem::path &raw_file,
                                                          const RawFileConfig &file_config) {
    return std::shared_ptr<Device>(DeviceDiscovery::open_raw_file(raw_file, file_config));
}

} // anonymous namespace

static HALGenericPythonBinder bind_connection_type([](auto &module) {
    py::enum_<ConnectionType>(module, "ConnectionType")
        .value("MIPI_LINK", ConnectionType::MIPI_LINK)
        .value("USB_LINK", ConnectionType::USB_LINK)
        .value("NETWORK_LINK", ConnectionType::NETWORK_LINK)
        .value("PROPRIETARY_LINK", ConnectionType::PROPRIETARY_LINK);
});

static HALClassPythonBinder<PluginCameraDescription> bind_plugin_camera_desc(
    [](auto &module, auto &class_binding) {
        class_binding.def_readonly("serial", &PluginCameraDescription::serial_)
            .def_readonly("connection", &PluginCameraDescription::connection_);
    },
    "PluginCameraDescription");

static HALClassPythonBinder<CameraDescription, PluginCameraDescription> bind_camera_desc(
    [](auto &module, auto &class_binding) {
        class_binding.def_readonly("integrator_name", &CameraDescription::integrator_name_)
            .def_readonly("plugin_name", &CameraDescription::plugin_name_);
    },
    "CameraDescription");

static HALGenericPythonBinder bind([](auto &module) {
    auto class_binding =
        py::class_<DeviceDiscovery>(module, "DeviceDiscovery", pybind_doc_hal["Metavision::DeviceDiscovery"])

            .def_static("open", &open_with_only_serial, py::return_value_policy::take_ownership, py::arg("serial"),
                        pybind_doc_hal["Metavision::DeviceDiscovery::open(const std::string &serial)"])
            .def_static(
                "open", &open_with_config_wrapper, py::return_value_policy::take_ownership, py::arg("serial"),
                py::arg("config"),
                pybind_doc_hal
                    ["Metavision::DeviceDiscovery::open(const std::string &serial, const DeviceConfig &config)"])
            .def_static(
                "open_raw_file", py::overload_cast<const std::filesystem::path &>(&open_raw_file_with_only_filename),
                py::return_value_policy::take_ownership, py::arg("raw_file"),
                pybind_doc_hal["Metavision::DeviceDiscovery::open_raw_file(const std::filesystem::path &raw_file)"])
            .def_static(
                "open_raw_file",
                py::overload_cast<const std::filesystem::path &, const RawFileConfig &>(
                    &open_raw_file_with_config_wrapper),
                py::return_value_policy::take_ownership, py::arg("raw_file"), py::arg("file_config"),
                pybind_doc_hal["Metavision::DeviceDiscovery::open_raw_file(const std::filesystem::path &raw_file, "
                               "const RawFileConfig &file_config)"])
            .def_static(
                "list",
                +[]() {
                    const auto l = DeviceDiscovery::list();
                    return std::vector<std::string>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list"])
            .def_static(
                "list_local",
                +[]() {
                    const auto l = DeviceDiscovery::list_local();
                    return std::vector<std::string>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list_local"])
            .def_static(
                "list_remote",
                +[]() {
                    const auto l = DeviceDiscovery::list_remote();
                    return std::vector<std::string>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list_remote"])
            .def_static(
                "list_available_sources",
                +[]() {
                    const auto l = DeviceDiscovery::list_available_sources();
                    return std::vector<CameraDescription>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list_available_sources"])
            .def_static(
                "list_available_sources_local",
                +[]() {
                    const auto l = DeviceDiscovery::list_available_sources_local();
                    return std::vector<CameraDescription>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list_available_sources_local"])
            .def_static(
                "list_available_sources_remote",
                +[]() {
                    const auto l = DeviceDiscovery::list_available_sources_remote();
                    return std::vector<CameraDescription>(l.begin(), l.end());
                },
                pybind_doc_hal["Metavision::DeviceDiscovery::list_available_sources_remote"])
            .def_static("list_device_config_options", &DeviceDiscovery::list_device_config_options, py::arg("serial"),
                        pybind_doc_hal["Metavision::DeviceDiscovery::list_device_config_options"]);
});

} // namespace Metavision
