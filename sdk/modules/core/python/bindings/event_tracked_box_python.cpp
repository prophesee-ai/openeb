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

#include "metavision/sdk/core/events/event_tracked_box.h"
#include "metavision/utils/pybind/pod_event_buffer.h"

namespace py = pybind11;

namespace Metavision {

void export_event_tracked_box(py::module &m) {
    PYBIND11_NUMPY_DTYPE(EventTrackedBox, t, x, y, w, h, class_id, track_id, class_confidence, tracking_confidence,
                         last_detection_update_time, nb_detections);
    py::array_t<EventTrackedBox> array;
    m.attr("EventTrackedBox") = array.dtype();

    Metavision::export_PODEventBuffer<EventTrackedBox>(m, "EventTrackedBoxBuffer");
}

} // namespace Metavision
