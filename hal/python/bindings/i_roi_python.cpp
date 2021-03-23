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
#include "metavision/hal/facilities/i_roi.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

std::vector<unsigned int> create_ROIs_wrapper(I_ROI &i_roi, py::list &roi_list) {
    std::vector<DeviceRoi> roi_vec;
    const ssize_t n_roi = py::len(roi_list);

    for (ssize_t roi_ind = 0; roi_ind < n_roi; ++roi_ind) {
        roi_vec.push_back((roi_list[roi_ind]).cast<DeviceRoi>());
    }

    return i_roi.create_ROIs(roi_vec);
}

void set_ROIs_from_bitword_wrapper(I_ROI &i_roi, py::list &roi_list, bool enable = true) {
    std::vector<unsigned int> roi_vec;
    const ssize_t n_roi = py::len(roi_list);

    for (ssize_t roi_ind = 0; roi_ind < n_roi; ++roi_ind) {
        roi_vec.push_back((roi_list[roi_ind]).cast<unsigned int>());
    }

    i_roi.set_ROIs_from_bitword(roi_vec, enable);
}

void set_ROIs_wrapper(I_ROI &i_roi, py::list &roi_list, bool enable = true) {
    std::vector<DeviceRoi> roi_vec;
    const ssize_t n_roi = py::len(roi_list);

    for (ssize_t roi_ind = 0; roi_ind < n_roi; ++roi_ind) {
        roi_vec.push_back((roi_list[roi_ind]).cast<DeviceRoi>());
    }

    i_roi.set_ROIs(roi_vec, enable);
}

void set_ROIs_cols_rows_wrapper(I_ROI &i_roi, py::list &cols_to_enable, py::list &rows_to_enable, bool enable = true) {
    std::vector<bool> cols_to_enable_vec;
    std::vector<bool> rows_to_enable_vec;
    const ssize_t n_cols = py::len(cols_to_enable);
    const ssize_t n_rows = py::len(rows_to_enable);

    for (ssize_t idx = 0; idx < n_cols; ++idx) {
        cols_to_enable_vec.push_back((cols_to_enable[idx]).cast<bool>());
    }

    for (ssize_t idx = 0; idx < n_rows; ++idx) {
        rows_to_enable_vec.push_back((rows_to_enable[idx]).cast<bool>());
    }

    i_roi.set_ROIs(cols_to_enable_vec, rows_to_enable_vec, enable);
}
} /* anonymous namespace */

static DeviceFacilityGetter<I_ROI> getter("get_i_roi");

static HALFacilityPythonBinder<I_ROI> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def("enable", &I_ROI::enable, py::arg("enable"), pybind_doc_hal["Metavision::I_ROI::enable"])
            .def("set_ROI", &I_ROI::set_ROI, py::arg("roi"), py::arg("enable") = true,
                 pybind_doc_hal["Metavision::I_ROI::set_ROI"])
            .def("set_ROIs_from_bitword", &I_ROI::set_ROIs_from_bitword, py::arg("vroiparams"),
                 py::arg("enable") = true, pybind_doc_hal["Metavision::I_ROI::set_ROIs_from_bitword"])
            .def("set_ROIs_from_bitword", &set_ROIs_from_bitword_wrapper, py::arg("roi_list"), py::arg("enable") = true)
            .def("set_ROIs", &set_ROIs_wrapper, py::arg("roi_list"), py::arg("enable") = true)
            .def("set_ROIs_from_file", &I_ROI::set_ROIs_from_file, py::arg("file_path"), py::arg("enable") = true,
                 pybind_doc_hal["Metavision::I_ROI::set_ROIs_from_file"])
            .def("set_ROIs", &set_ROIs_cols_rows_wrapper, py::arg("cols_to_enable"), py::arg("rows_to_enable"),
                 py::arg("enable") = true)
            .def("create_ROI", &I_ROI::create_ROI, py::arg("roi"), pybind_doc_hal["Metavision::I_ROI::create_ROI"])
            .def("create_ROIs", &create_ROIs_wrapper, py::arg("roi_list"));
    },
    "I_ROI", pybind_doc_hal["Metavision::I_ROI"]);

} // namespace Metavision
