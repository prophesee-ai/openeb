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

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

void write_register_wrapper1(I_HW_Register &self, uint32_t address, uint32_t v) {
    self.write_register(address, v);
}

void write_register_wrapper2(I_HW_Register &self, const std::string &address, uint32_t v) {
    self.write_register(address, v);
}

void write_register_wrapper3(I_HW_Register &self, const std::string &address, const std::string &bitfield, uint32_t v) {
    self.write_register(address, bitfield, v);
}

uint32_t read_register_wrapper1(I_HW_Register &self, uint32_t address) {
    return self.read_register(address);
}

uint32_t read_register_wrapper2(I_HW_Register &self, const std::string &address) {
    return self.read_register(address);
}

uint32_t read_register_wrapper3(I_HW_Register &self, const std::string &address, const std::string &bitfield) {
    return self.read_register(address, bitfield);
}

} /* anonymous namespace */

static DeviceFacilityGetter<I_HW_Register> getter("get_i_hw_register");

static HALFacilityPythonBinder<I_HW_Register> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("write_register", &write_register_wrapper1, py::arg("address"), py::arg("v"),
                 pybind_doc_hal["Metavision::I_HW_Register::write_register(uint32_t address, uint32_t v)=0"])
            .def("write_register", &write_register_wrapper2, py::arg("address"), py::arg("v"),
                 pybind_doc_hal["Metavision::I_HW_Register::write_register(const std::string &address, uint32_t v)=0"])
            .def("write_register", &write_register_wrapper3, py::arg("address"), py::arg("bitfield"), py::arg("v"),
                 pybind_doc_hal["Metavision::I_HW_Register::write_register(const std::string &address, const "
                                "std::string &bitfield, uint32_t v)=0"])
            .def("read_register", &read_register_wrapper1, py::arg("address"),
                 pybind_doc_hal["Metavision::I_HW_Register::read_register(uint32_t address)=0"])
            .def("read_register", &read_register_wrapper2, py::arg("address"),
                 pybind_doc_hal["Metavision::I_HW_Register::read_register(const std::string &address)=0"])
            .def("read_register", &read_register_wrapper3, py::arg("address"), py::arg("bitfield"),
                 pybind_doc_hal["Metavision::I_HW_Register::read_register(const std::string &address, const "
                                "std::string &bitfield)=0"]);
    },
    "I_HW_Register", pybind_doc_hal["Metavision::I_HW_Register"]);

} // namespace Metavision
