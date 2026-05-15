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
#include "metavision/sdk/core/algorithms/adaptive_rate_events_splitter_algorithm.h"
#include "pb_doc_core.h"

namespace Metavision {

bool AdaptiveRateEventsSplitter_process_events_array(Metavision::AdaptiveRateEventsSplitterAlgorithm &evsplitter,
                                                     const py::array_t<Metavision::EventCD> &events) {
    auto info_events = events.request();
    if (info_events.ndim != 1) {
        throw std::runtime_error("Wrong events dim");
    }
    auto events_ptr     = static_cast<Metavision::EventCD *>(info_events.ptr);
    const int nb_events = info_events.shape[0];

    return evsplitter.process_events(events_ptr, events_ptr + nb_events);
}

bool AdaptiveRateEventsSplitter_process_events_buffer(
    Metavision::AdaptiveRateEventsSplitterAlgorithm &evsplitter,
    const Metavision::PODEventBuffer<Metavision::EventCD> &buff_events) {
    return evsplitter.process_events(buff_events.buffer_.cbegin(), buff_events.buffer_.cend());
}

void AdaptiveRateEventsSplitter_retrieve_events_in_podeventbuffer(
    Metavision::AdaptiveRateEventsSplitterAlgorithm &evsplitter, PODEventBuffer<EventCD> &out) {
    evsplitter.retrieve_events(out.buffer_);
}

void export_adaptive_rate_events_splitter_algorithm(py::module &m) {
    py::class_<AdaptiveRateEventsSplitterAlgorithm>(m, "AdaptiveRateEventsSplitterAlgorithm",
                                                    pybind_doc_core["Metavision::AdaptiveRateEventsSplitterAlgorithm"])
        .def(py::init<int, int, float, int>(), py::arg("height"), py::arg("width"), py::arg("thr_var_per_event") = 5e-4,
             py::arg("downsampling_factor") = 2,
             pybind_doc_core["Metavision::AdaptiveRateEventsSplitterAlgorithm::AdaptiveRateEventsSplitterAlgorithm"])
        .def("process_events", &AdaptiveRateEventsSplitter_process_events_array, py::arg("events_np"),
             "Takes a chunk of events (numpy array of EventCD) and updates the internal state of the EventsSplitter. "
             "Returns True if the frame is ready, False otherwise.")
        .def("process_events", &AdaptiveRateEventsSplitter_process_events_buffer, py::arg("events_buf"),
             "Takes a chunk of events (EventCDBuffer) and updates the internal state of the EventsSplitter. Returns "
             "True if the frame is ready, False otherwise.")
        .def("retrieve_events", &AdaptiveRateEventsSplitter_retrieve_events_in_podeventbuffer, py::arg("events_buf"),
             "Retrieves the events (EventCDBuffer) and reinitializes the state of the EventsSplitter.")
        .def_static("get_empty_output_buffer", &getEmptyPODBuffer<EventCD>, doc_get_empty_output_buffer_str);
}

} // namespace Metavision
