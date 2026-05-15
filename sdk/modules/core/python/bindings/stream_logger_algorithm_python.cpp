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
#include <pybind11/functional.h>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/utils/pybind/async_algorithm_process_helper.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"
#include "pb_doc_core.h"

namespace py = pybind11;

namespace Metavision {

void export_stream_logger_algorithm(py::module &m) {
    py::class_<StreamLoggerAlgorithm>(m, "StreamLoggerAlgorithm", pybind_doc_core["Metavision::StreamLoggerAlgorithm"])
        .def(py::init<std::string, std::int32_t, std::int32_t>(), py::arg("filename"), py::arg("width"),
             py::arg("height"), pybind_doc_core["Metavision::StreamLoggerAlgorithm::StreamLoggerAlgorithm"])
        .def("enable", &StreamLoggerAlgorithm::enable, py::arg("state"), py::arg("reset_ts") = true,
             py::arg("split_time_seconds") = -1, pybind_doc_core["Metavision::StreamLoggerAlgorithm::enable"])
        .def("is_enable", &StreamLoggerAlgorithm::is_enable,
             pybind_doc_core["Metavision::StreamLoggerAlgorithm::is_enable"])
        .def("get_split_time_seconds", &StreamLoggerAlgorithm::get_split_time_seconds)
        .def("change_destination", &StreamLoggerAlgorithm::change_destination, py::arg("filename"),
             py::arg("reset_ts") = true)
        .def("process_events", &process_events_array_ts_async<StreamLoggerAlgorithm, EventCD>, py::arg("events_np"),
             py::arg("ts"), doc_process_events_array_ts_async_str)
        .def(
            "get_native_process_cd_events_callback",
            +[](StreamLoggerAlgorithm &self) {
                return std::function<void(const EventCD *, const EventCD *)>(
                    [&self](const EventCD *begin, const EventCD *end) {
                        if (begin < end) {
                            self.process_events(begin, end, (end - 1)->t);
                        }
                    });
            },
            "Returns a callback to be passed to the event_cd decoder from Metavision HAL.")
        .def(
            "get_native_process_ext_trigger_events_callback",
            +[](StreamLoggerAlgorithm &self) {
                return std::function<void(const EventExtTrigger *, const EventExtTrigger *)>(
                    [&self](const EventExtTrigger *begin, const EventExtTrigger *end) {
                        if (begin < end) {
                            self.process_events(begin, end, (end - 1)->t);
                        }
                    });
            },
            "Returns a callback to be passed to the event_ext_trigger decoder from Metavision HAL.")

        .def("close", &StreamLoggerAlgorithm::close);
}

} // namespace Metavision
