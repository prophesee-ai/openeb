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

#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/utils/pybind/pod_event_buffer.h"

namespace py = pybind11;

namespace Metavision {

void export_event_ext_trigger(py::module &m) {
    PYBIND11_NUMPY_DTYPE(Metavision::EventExtTrigger, p, t, id);
    py::array_t<Metavision::EventExtTrigger> array;
    m.attr("EventExtTrigger") = array.dtype();

    py::class_<Metavision::EventExtTrigger>(m, "_EventExtTrigger_decode")
        .def(py::init<short, Metavision::timestamp, short>());

    Metavision::export_PODEventBuffer<Metavision::EventExtTrigger>(m, "EventExtTriggerBuffer");
}

} // namespace Metavision
