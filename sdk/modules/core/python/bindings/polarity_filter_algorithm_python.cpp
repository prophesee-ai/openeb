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
#include "metavision/sdk/core/algorithms/polarity_filter_algorithm.h"
#include "pb_doc_core.h"

namespace Metavision {

void export_polarity_filter_algorithm(py::module &m) {
    py::class_<PolarityFilterAlgorithm>(m, "PolarityFilterAlgorithm",
                                        pybind_doc_core["Metavision::PolarityFilterAlgorithm"])
        .def(py::init<std::int16_t>(), py::arg("polarity") = 0,
             pybind_doc_core["Metavision::PolarityFilterAlgorithm::PolarityFilterAlgorithm"])
        .def("process_events", &process_events_array_sync<PolarityFilterAlgorithm, EventCD>, py::arg("input_np"),
             py::arg("output_buf"), doc_process_events_array_sync_str)
        .def("process_events", &process_events_buffer_sync<PolarityFilterAlgorithm, EventCD>, py::arg("input_buf"),
             py::arg("output_buf"), doc_process_events_buffer_sync_str)
        .def("process_events_", &process_events_buffer_sync_inplace<PolarityFilterAlgorithm, EventCD>,
             py::arg("events_buf"), doc_process_events_buffer_sync_inplace_str)
        .def_static("get_empty_output_buffer", &getEmptyPODBuffer<EventCD>, doc_get_empty_output_buffer_str)
        .def_property("polarity", &PolarityFilterAlgorithm::polarity, &PolarityFilterAlgorithm::set_polarity);
}

} // namespace Metavision
