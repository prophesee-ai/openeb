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

#include <pybind11/numpy.h>

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

void decode_wrapper(I_EventsStreamDecoder *decoder, py::array_t<I_EventsStream::RawData> *array) {
    py::buffer_info buf = array->request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Decoder need a array of events the provided buffer as " + std::to_string(buf.ndim) +
                                 " dimensions");
    }
    decoder->decode((I_EventsStream::RawData *)buf.ptr, (I_EventsStream::RawData *)buf.ptr + buf.size);
}

} /* anonymous namespace */

static DeviceFacilityGetter<I_EventsStreamDecoder> getter("get_i_events_stream_decoder");

static HALFacilityPythonBinder<I_EventsStreamDecoder> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("get_last_timestamp", &I_EventsStreamDecoder::get_last_timestamp,
                 pybind_doc_hal["Metavision::I_EventsStreamDecoder::get_last_timestamp"])
            .def("decode", &decode_wrapper, py::arg("RawData"),
                 "Decodes raw data. Identifies the events in the buffer and dispatches it to the instance "
                 "Event Decoder corresponding to each event type\n"
                 "\n"
                 "Args:\n"
                 "    RawData: Numpy array of Events\n")
            .def(
                "add_time_callback",
                +[](I_EventsStreamDecoder &self, py::object object) {
                    std::function<void(timestamp base_time)> gil_cb = [=](timestamp base_time) {
                        py::gil_scoped_acquire acquire;
                        object(base_time);
                    };
                    self.add_time_callback(gil_cb);
                },
                py::arg("cb"), pybind_doc_hal["Metavision::I_EventsStreamDecoder::add_time_callback"])
            .def("remove_callback", &I_EventsStreamDecoder::remove_time_callback, py::arg("callback_id"),
                 pybind_doc_hal["Metavision::I_EventsStreamDecoder::remove_time_callback"]);
    },
    "I_Decoder", pybind_doc_hal["Metavision::I_EventsStreamDecoder"]);

} // namespace Metavision
