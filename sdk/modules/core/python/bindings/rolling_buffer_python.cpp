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
#include "metavision/sdk/core/algorithms/roi_filter_algorithm.h"
#include "metavision/sdk/core/utils/rolling_event_buffer.h"
#include "metavision/utils/pybind/sync_algorithm_process_helper.h"
#include "metavision/utils/pybind/rolling_event_buffer.h"

#include "pb_doc_core.h"

namespace Metavision {

void export_rolling_event_cd_buffer(py::module &m) {
    py::enum_<RollingEventBufferMode>(m, "RollingEventBufferMode")
        .value("N_US", RollingEventBufferMode::N_US)
        .value("N_EVENTS", RollingEventBufferMode::N_EVENTS);

    py::class_<RollingEventBufferConfig>(m, "RollingEventBufferConfig", py::module_local())
        .def(py::init())
        .def_static("make_n_us", &RollingEventBufferConfig::make_n_us, py::return_value_policy::take_ownership,
                    py::arg(""), pybind_doc_core["Metavision::RollingEventBufferConfig::make_n_us"])
        .def_static("make_n_events", &RollingEventBufferConfig::make_n_events, py::return_value_policy::take_ownership,
                    py::arg(""), pybind_doc_core["Metavision::RollingEventBufferConfig::make_n_events"])
        .def_readwrite("mode", &RollingEventBufferConfig::mode)
        .def_readwrite("delta_ts", &RollingEventBufferConfig::delta_ts)
        .def_readwrite("delta_n_events", &RollingEventBufferConfig::delta_n_events);

    export_rolling_event_buffer<EventCD>(m, "RollingEventCDBuffer", pybind_doc_core);
}
} // namespace Metavision