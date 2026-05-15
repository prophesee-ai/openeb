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
#include <pybind11/functional.h>

#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "hal_python_binder.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "pb_doc_hal.h"

namespace py = pybind11;

namespace Metavision {

static DeviceFacilityGetter<I_EventDecoder<EventCD>> getter("get_i_event_cd_decoder");

static HALFacilityPythonBinder<I_EventDecoder<EventCD>> bind_decoder(
    [](auto &module, auto &class_binding) {
        using EventCDIterator_t = I_EventDecoder<EventCD>::EventIterator_t;

        class_binding
            .def(
                "add_event_buffer_callback",
                +[](I_EventDecoder<EventCD> &self, py::object object) {
                    std::function<void(EventCDIterator_t begin, EventCDIterator_t end)> gil_cb =
                        [=](EventCDIterator_t begin, EventCDIterator_t end) {
                            // py::array make a copy of the data
                            auto py_array = py::array(end - begin, begin);
                            py::gil_scoped_acquire acquire;
                            object(py_array);
                        };
                    return self.add_event_buffer_callback(gil_cb);
                },
                pybind_doc_hal["Metavision::I_EventDecoder::add_event_buffer_callback"])
            .def("remove_callback", &I_EventDecoder<EventCD>::remove_callback,
                 pybind_doc_hal["Metavision::I_EventDecoder::remove_callback"])
            .def(
                "add_event_buffer_nocopy_callback",
                +[](I_EventDecoder<EventCD> &self, py::object object) {
                    std::function<void(EventCDIterator_t begin, EventCDIterator_t end)> gil_cb =
                        [=](EventCDIterator_t begin, EventCDIterator_t end) {
                            auto capsule  = py::capsule(begin, [](void *v) {});
                            auto py_array = py::array_t<EventCD>(end - begin, begin, capsule);

                            py::gil_scoped_acquire acquire;
                            object(py_array);
                        };
                    return self.add_event_buffer_callback(gil_cb);
                })
            .def(
                "add_event_buffer_native_callback",
                +[](I_EventDecoder<EventCD> &self, std::function<void(const EventCD *, const EventCD *)> fun) {
                    return self.add_event_buffer_callback(fun);
                },
                "Pass a native function as event callback (faster than a python one).\n",
                "\nIts signature has to be `void function(const Metavision::EventCD * begin,",
                "const Metavision::EventCD * end)`.");
    },
    "I_EventDecoder_EventCD");

} // namespace Metavision
