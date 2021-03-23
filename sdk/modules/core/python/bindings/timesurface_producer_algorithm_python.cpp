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
#include "metavision/utils/pybind/async_algorithm_process_helper.h"
#include "metavision/sdk/core/algorithms/time_surface_producer_algorithm.h"
#include "pb_doc_core.h"

namespace Metavision {

void export_timesurface_producer_algorithm(py::module &m) {
    using TimeSurfaceProducerAlgorithmMergePolarities = TimeSurfaceProducerAlgorithm<1>;

    const std::string tsa = pybind_doc_core["Metavision::TimeSurfaceProducerAlgorithm"];
    const std::string ts_doc_merge_pol_end =
        "This timesurface contains only one channel (events with different polarities are stored all together "
        "in the same channel).\n"
        "To use separate channels for polarities, use TimeSurfaceProducerAlgorithmSplitPolarities instead\n";
    const std::string ts_doc_mege_pol_full = tsa + "\n\n" + ts_doc_merge_pol_end;

    py::class_<TimeSurfaceProducerAlgorithmMergePolarities>(m, "TimeSurfaceProducerAlgorithmMergePolarities",
                                                            ts_doc_mege_pol_full.c_str())
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"),
             pybind_doc_core["Metavision::TimeSurfaceProducerAlgorithm::TimeSurfaceProducerAlgorithm"])
        .def(
            "set_output_callback",
            [](TimeSurfaceProducerAlgorithmMergePolarities &algo, const py::object &object) {
                TimeSurfaceProducerAlgorithmMergePolarities::OutputCb cb =
                    [object](timestamp ts, const MostRecentTimestampBuffer &most_recent_timestamp_buffer) {
                        assert(most_recent_timestamp_buffer.channels() == 1);
                        object(ts, most_recent_timestamp_buffer);
                    };
                algo.set_output_callback(cb);
            },
            "Sets a callback to retrieve the produced time surface")
        .def("process_events", &process_events_array_async<TimeSurfaceProducerAlgorithmMergePolarities, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str);

    using TimeSurfaceProducerAlgorithmSplitPolarities = TimeSurfaceProducerAlgorithm<2>;

    const std::string ts_doc_split_pol_end =
        "This timesurface contains two channels (events with different polarities are stored in separate channels\n"
        "To use single channel, use TimeSurfaceProducerAlgorithmMergePolarities instead\n";
    const std::string ts_doc_split_pol_full = tsa + "\n\n" + ts_doc_split_pol_end;

    py::class_<TimeSurfaceProducerAlgorithmSplitPolarities>(m, "TimeSurfaceProducerAlgorithmSplitPolarities",
                                                            ts_doc_split_pol_full.c_str())
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"),
             pybind_doc_core["Metavision::TimeSurfaceProducerAlgorithm::TimeSurfaceProducerAlgorithm"])
        .def(
            "set_output_callback",
            [](TimeSurfaceProducerAlgorithmSplitPolarities &algo, const py::object &object) {
                TimeSurfaceProducerAlgorithmSplitPolarities::OutputCb cb =
                    [object](timestamp ts, const MostRecentTimestampBuffer &most_recent_timestamp_buffer) {
                        assert(most_recent_timestamp_buffer.channels() == 2);
                        object(ts, most_recent_timestamp_buffer);
                    };
                algo.set_output_callback(cb);
            },
            "Sets a callback to retrieve the produced time surface")
        .def("process_events", &process_events_array_async<TimeSurfaceProducerAlgorithmSplitPolarities, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str);
}

} // namespace Metavision
