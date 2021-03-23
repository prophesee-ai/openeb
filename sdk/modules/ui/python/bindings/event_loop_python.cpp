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

#include "metavision/sdk/ui/utils/event_loop.h"

#include "pb_doc_ui.h"

namespace py = pybind11;

namespace Metavision {

void export_event_loop(py::module &m) {
    using namespace pybind11::literals;

    py::class_<EventLoop>(m, "EventLoop", pybind_doc_ui["Metavision::EventLoop"])
        .def(py::init<>())
        .def_static("poll_and_dispatch", &EventLoop::poll_and_dispatch, "sleep_time_ms"_a = 0,
                    pybind_doc_ui["Metavision::EventLoop::poll_and_dispatch"]);
}

} // namespace Metavision