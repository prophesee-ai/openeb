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

#include <pybind11/stl.h>

#include "hal_python_binder.h"
#include "metavision/hal/utils/device_config.h"
#include "pb_doc_hal.h"

namespace Metavision {

static HALClassPythonBinder<DeviceConfig> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def(py::init())
            .def(py::init<const DeviceConfig &>())
            .def_static("get_format_key", &DeviceConfig::get_format_key,
                        pybind_doc_hal["Metavision::DeviceConfig::get_format_key"])
            .def("format", &DeviceConfig::format, pybind_doc_hal["Metavision::DeviceConfig::format"])
            .def("set_format", &DeviceConfig::set_format, py::arg("format"),
                 pybind_doc_hal["Metavision::DeviceConfig::set_format"])
            .def_static("get_biases_range_check_bypass_key", &DeviceConfig::get_biases_range_check_bypass_key,
                        pybind_doc_hal["Metavision::DeviceConfig::get_biases_range_check_bypass_key"])
            .def("biases_range_check_bypass", &DeviceConfig::biases_range_check_bypass,
                 pybind_doc_hal["Metavision::DeviceConfig::biases_range_check_bypass"])
            .def("enable_biases_range_check_bypass", &DeviceConfig::enable_biases_range_check_bypass,
                 py::arg("enabled"), pybind_doc_hal["Metavision::DeviceConfig::enable_biases_range_check_bypass"])
            .def("get",
                 static_cast<std::string (DeviceConfig::*)(const std::string &, const std::string &) const>(
                     &DeviceConfig::get),
                 py::arg("key"), py::arg("def") = "",
                 pybind_doc_hal["Metavision::DeviceConfig::get(const std::string &key, const std::string "
                                "&def=std::string()) const"])
            .def("get_bool",
                 static_cast<bool (DeviceConfig::*)(const std::string &, const bool &) const>(&DeviceConfig::get),
                 py::arg("key"), py::arg("def") = true,
                 pybind_doc_hal["Metavision::DeviceConfig::get(const std::string &key, const T &def=T()) const"])
            .def("get_int",
                 static_cast<int (DeviceConfig::*)(const std::string &, const int &) const>(&DeviceConfig::get),
                 py::arg("key"), py::arg("def") = 0,
                 pybind_doc_hal["Metavision::DeviceConfig::get(const std::string &key, const T &def=T()) const"])
            .def("get_double",
                 static_cast<double (DeviceConfig::*)(const std::string &, const double &) const>(&DeviceConfig::get),
                 py::arg("key"), py::arg("def") = 0.0,
                 pybind_doc_hal["Metavision::DeviceConfig::get(const std::string &key, const T &def=T()) const"])
            .def("set", static_cast<void (DeviceConfig::*)(const std::string &, bool)>(&DeviceConfig::set),
                 py::arg("key"), py::arg("value"),
                 pybind_doc_hal["Metavision::DeviceConfig::set(const std::string &key, bool value)"])
            .def("set",
                 static_cast<void (DeviceConfig::*)(const std::string &, const std::string &)>(&DeviceConfig::set),
                 py::arg("key"), py::arg("value"),
                 "Sets a value for a named key in the config dictionary This is an overloaded member function, "
                 "provided for convenience. It differs from the above function only in what argument(s) it accepts.")
            .def("set", &DeviceConfig::set<int>, py::arg("key"), py::arg("value"),
                 "Sets a value for a named key in the config dictionary This is an overloaded member function, "
                 "provided for convenience. It differs from the above function only in what argument(s) it accepts.")
            .def("set", &DeviceConfig::set<double>, py::arg("key"), py::arg("value"),
                 "Sets a value for a named key in the config dictionary This is an overloaded member function, "
                 "provided for convenience. It differs from the above function only in what argument(s) it accepts.");
    },
    "DeviceConfig", pybind_doc_hal["Metavision::DeviceConfig"]);

py::object get_range_wrapper(DeviceConfigOption &self) {
    switch (self.type()) {
    case DeviceConfigOption::Type::Int:
        return py::cast(self.get_range<int>());
    case DeviceConfigOption::Type::Double:
        return py::cast(self.get_range<double>());
    default:
        break;
    }
    throw std::runtime_error("get_range called with incompatible type");
}

py::object get_default_value_wrapper(DeviceConfigOption &self) {
    switch (self.type()) {
    case DeviceConfigOption::Type::Boolean:
        return py::cast(self.get_default_value<bool>());
    case DeviceConfigOption::Type::Int:
        return py::cast(self.get_default_value<int>());
    case DeviceConfigOption::Type::Double:
        return py::cast(self.get_default_value<double>());
    case DeviceConfigOption::Type::String:
        return py::cast(self.get_default_value<std::string>());
    default:
        break;
    }
    throw std::runtime_error("get_range called with incompatible type");
}

static HALClassPythonBinder<DeviceConfigOption> bind2(
    [](auto &module, auto &class_binding) {
        py::enum_<DeviceConfigOption::Type>(class_binding, "Type", py::arithmetic())
            .value("INVALID", DeviceConfigOption::Type::Invalid)
            .value("BOOLEAN", DeviceConfigOption::Type::Boolean)
            .value("INT", DeviceConfigOption::Type::Int)
            .value("DOUBLE", DeviceConfigOption::Type::Double)
            .value("STRING", DeviceConfigOption::Type::String);

        class_binding.def(py::init())
            .def(py::init<bool>())
            .def(py::init<int, int, int>())
            .def(py::init<double, double, double>())
            .def(py::init<const std::vector<std::string> &, const std::string &>())
            .def(py::init<const DeviceConfigOption &>())
            .def("get_range", &get_range_wrapper, pybind_doc_hal["Metavision::DeviceConfigOption::get_range"])
            .def("get_values", &DeviceConfigOption::get_values,
                 pybind_doc_hal["Metavision::DeviceConfigOption::get_values"])
            .def("get_default_value", &get_default_value_wrapper,
                 pybind_doc_hal["Metavision::DeviceConfigOption::get_default_value"])
            .def("type", &DeviceConfigOption::type, pybind_doc_hal["Metavision::DeviceConfigOption::type"]);
    },
    "DeviceConfigOption", pybind_doc_hal["Metavision::DeviceConfigOption"]);

} // namespace Metavision
