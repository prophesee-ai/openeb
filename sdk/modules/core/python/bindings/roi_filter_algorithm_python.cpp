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

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/pybind/sync_algorithm_process_helper.h"
#include "metavision/sdk/core/algorithms/roi_filter_algorithm.h"
#include "pb_doc_core.h"

namespace Metavision {

void export_roi_filter_algorithm(py::module &m) {
    py::class_<RoiFilterAlgorithm>(m, "RoiFilterAlgorithm", pybind_doc_core["Metavision::RoiFilterAlgorithm"])
        .def(py::init<std::int32_t, std::int32_t, std::int32_t, std::int32_t, bool>(), py::arg("x0"), py::arg("y0"),
             py::arg("x1"), py::arg("y1"), py::arg("output_relative_coordinates") = false,
             pybind_doc_core["Metavision::RoiFilterAlgorithm::RoiFilterAlgorithm"])
        .def("process_events", &process_events_array_sync<RoiFilterAlgorithm, EventCD>, py::arg("input_np"),
             py::arg("output_buf"), doc_process_events_array_sync_str)
        .def("process_events", &process_events_buffer_sync<RoiFilterAlgorithm, EventCD>, py::arg("input_buf"),
             py::arg("output_buf"), doc_process_events_buffer_sync_str)
        .def("process_events_", &process_events_buffer_sync_inplace<RoiFilterAlgorithm, EventCD>, py::arg("events_buf"),
             doc_process_events_buffer_sync_inplace_str)
        .def_static("get_empty_output_buffer", &getEmptyPODBuffer<EventCD>, doc_get_empty_output_buffer_str)
        .def_property("x0", &RoiFilterAlgorithm::x0, &RoiFilterAlgorithm::set_x0,
                      pybind_doc_core["Metavision::RoiFilterAlgorithm::x0"])
        .def_property("y0", &RoiFilterAlgorithm::y0, &RoiFilterAlgorithm::set_y0,
                      pybind_doc_core["Metavision::RoiFilterAlgorithm::y0"])
        .def_property("x1", &RoiFilterAlgorithm::x1, &RoiFilterAlgorithm::set_x1,
                      pybind_doc_core["Metavision::RoiFilterAlgorithm::x1"])
        .def_property("y1", &RoiFilterAlgorithm::y1, &RoiFilterAlgorithm::set_y1,
                      pybind_doc_core["Metavision::RoiFilterAlgorithm::y1"])
        .def("is_resetting", &RoiFilterAlgorithm::is_resetting,
             pybind_doc_core["Metavision::RoiFilterAlgorithm::is_resetting"]);
}

} // namespace Metavision
