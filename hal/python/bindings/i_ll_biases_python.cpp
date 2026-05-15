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
#include "metavision/hal/facilities/i_ll_biases.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

py::dict get_all_biases_wrapper(I_LL_Biases &self) {
    auto all_biases = self.get_all_biases();
    py::dict dictionary;
    for (auto it = all_biases.begin(), it_end = all_biases.end(); it != it_end; ++it) {
        dictionary[py::str(it->first)] = it->second;
    }
    return dictionary;
}

LL_Bias_Info get_bias_info_wrapper(I_LL_Biases &self, const std::string &bias_name) {
    LL_Bias_Info info;
    self.get_bias_info(bias_name, info);
    return info;
}

} /* anonymous namespace */

static DeviceFacilityGetter<I_LL_Biases> getter("get_i_ll_biases");

static HALFacilityPythonBinder<I_LL_Biases> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("set", &I_LL_Biases::set, py::arg("bias_name"), py::arg("bias_value"),
                 pybind_doc_hal["Metavision::I_LL_Biases::set"])
            .def("get", &I_LL_Biases::get, py::arg("bias_name"), pybind_doc_hal["Metavision::I_LL_Biases::get"])
            .def("get_all_biases", &get_all_biases_wrapper, pybind_doc_hal["Metavision::I_LL_Biases::get_all_biases"])
            .def("get_bias_info", &get_bias_info_wrapper, py::arg("bias_name"),
                 "Gets bias metadata.\n\n"
                 "    :bias_name: Name of the bias whose metadata to get\n\n"
                 "    :return: Metadata of the bias");
    },
    "I_LL_Biases", pybind_doc_hal["Metavision::I_LL_Biases"]);

static HALFacilityPythonBinder<LL_Bias_Info> bind2(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("get_description", &LL_Bias_Info::get_description,
                 pybind_doc_hal["Metavision::LL_Bias_Info::get_description"])
            .def("get_category", &LL_Bias_Info::get_category, pybind_doc_hal["Metavision::LL_Bias_Info::get_category"])
            .def("get_bias_range", &LL_Bias_Info::get_bias_range,
                 pybind_doc_hal["Metavision::LL_Bias_Info::get_bias_range"])
            .def("get_bias_recommended_range", &LL_Bias_Info::get_bias_recommended_range,
                 pybind_doc_hal["Metavision::LL_Bias_Info::get_bias_recommended_range"])
            .def("get_bias_allowed_range", &LL_Bias_Info::get_bias_allowed_range,
                 pybind_doc_hal["Metavision::LL_Bias_Info::get_bias_allowed_range"])
            .def("is_modifiable", &LL_Bias_Info::is_modifiable,
                 pybind_doc_hal["Metavision::LL_Bias_Info::is_modifiable"]);
    },
    "LL_Bias_Info", pybind_doc_hal["Metavision::LL_Bias_Info"]);

} // namespace Metavision
