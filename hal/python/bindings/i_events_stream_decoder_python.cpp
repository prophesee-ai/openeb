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
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "pb_doc_hal.h"

// Needed to avoid copies of vectors of RawData
PYBIND11_MAKE_OPAQUE(std::vector<Metavision::I_EventsStream::RawData>);

namespace Metavision {

namespace {

void decode_wrapper(I_EventsStreamDecoder *decoder, py::object &raw_buffer) {
    decoder->decode(raw_buffer.cast<DataTransfer::BufferPtr &>());
}

// Provide this overload to avoid having to convert opaque vectors to py::array_t
void decode_wrapper_vector(I_EventsStreamDecoder *decoder, std::vector<I_EventsStream::RawData> &buffer) {
    decoder->decode(buffer.data(), buffer.data() + buffer.size());
}

} /* anonymous namespace */

static DeviceFacilityGetter<I_EventsStreamDecoder> getter("get_i_events_stream_decoder");

static HALFacilityPythonBinder<I_EventsStreamDecoder> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("get_last_timestamp", &I_EventsStreamDecoder::get_last_timestamp,
                 pybind_doc_hal["Metavision::I_EventsStreamDecoder::get_last_timestamp"])
            .def("decode", &decode_wrapper,
                 "Decodes raw data. Identifies the events in the buffer and dispatches it to the instance "
                 "Event Decoder corresponding to each event type\n"
                 "\n"
                 "Args:\n"
                 "    RawData: Numpy array of Events\n")
            .def("decode", &decode_wrapper_vector, py::arg("RawData"),
                 "Decodes raw data. Identifies the events in the buffer and dispatches it to the instance "
                 "Event Decoder corresponding to each event type\n"
                 "\n"
                 "Args:\n"
                 "    RawData: Array of events\n")
            .def(
                "add_time_callback",
                +[](I_EventsStreamDecoder &self, py::object object) {
                    std::function<void(timestamp base_time)> gil_cb = [=](timestamp base_time) {
                        py::gil_scoped_acquire acquire;
                        object(base_time);
                    };
                    return self.add_time_callback(gil_cb);
                },
                py::arg("cb"), pybind_doc_hal["Metavision::I_EventsStreamDecoder::add_time_callback"])
            .def("remove_callback", &I_EventsStreamDecoder::remove_time_callback, py::arg("callback_id"),
                 pybind_doc_hal["Metavision::I_EventsStreamDecoder::remove_time_callback"]);
    },
    "I_Decoder", pybind_doc_hal["Metavision::I_EventsStreamDecoder"]);

} // namespace Metavision
