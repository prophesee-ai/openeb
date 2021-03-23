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
#include "metavision/utils/pybind/pod_event_buffer.h"

namespace py = pybind11;

namespace Metavision {

void export_event_cd(py::module &m) {
    PYBIND11_NUMPY_DTYPE(Metavision::Event2d, x, y, p, t);

    PYBIND11_NUMPY_DTYPE(Metavision::EventCD, x, y, p, t);
    py::array_t<Metavision::EventCD> array;
    m.attr("EventCD") = array.dtype();

    py::class_<Metavision::EventCD>(m, "_EventCD_decode")
        .def(py::init<unsigned short, unsigned short, short, Metavision::timestamp>());

    Metavision::export_PODEventBuffer<Metavision::EventCD>(m, "EventCDBuffer");
}

} // namespace Metavision
