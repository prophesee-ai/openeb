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

#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "hal_python_binder.h"

#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"

#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/base/events/raw_event_frame_histo.h"
#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {

void decode_wrapper_diff(I_EventFrameDecoder<RawEventFrameDiff> *decoder, py::object &raw_buffer) {
    decoder->decode(py::cast<DataTransfer::BufferPtr>(raw_buffer));
}

void decode_wrapper_histo(I_EventFrameDecoder<RawEventFrameHisto> *decoder, py::object &raw_buffer) {
    decoder->decode(py::cast<DataTransfer::BufferPtr>(raw_buffer));
}

} /* anonymous namespace */

static DeviceFacilityGetter<I_EventFrameDecoder<RawEventFrameDiff>> getter_diff("get_i_event_frame_diff_decoder");

static DeviceFacilityGetter<I_EventFrameDecoder<RawEventFrameHisto>> getter_histo("get_i_event_frame_histo_decoder");

static HALFacilityPythonBinder<I_EventFrameDecoder<RawEventFrameDiff>> bind_decoder_diff(
    [](auto &module, auto &class_binding) {
        class_binding
            .def(
                "add_event_frame_callback",
                +[](I_EventFrameDecoder<RawEventFrameDiff> &self, py::object object) {
                    std::function<void(const RawEventFrameDiff &)> gil_cb = [=](const RawEventFrameDiff &frame) {
                        // Wrap memory space as a python readable buffer.
                        py::gil_scoped_acquire acquire;
                        object(frame);
                    };
                    return self.add_event_frame_callback(gil_cb);
                },
                pybind_doc_hal["Metavision::I_EventFrameDecoder::add_event_frame_callback"])
            .def("decode", &decode_wrapper_diff, py::arg("RawData"),
                 "Decodes raw data. Identifies the events in the buffer and dispatches it to the instance "
                 "Event Decoder corresponding to each event type\n"
                 "\n"
                 "Args:\n"
                 "    RawData: Numpy array of Event Frames\n")
            .def("get_height", &I_EventFrameDecoder<RawEventFrameDiff>::get_height, "get height")
            .def("get_width", &I_EventFrameDecoder<RawEventFrameDiff>::get_width, "get width")
            .def("remove_callback", &I_EventFrameDecoder<RawEventFrameDiff>::remove_callback,
                 pybind_doc_hal["Metavision::I_EventFrameDecoder::remove_callback"]);
    },
    "I_EventFrameDecoder_RawEventFrameDiff");

static HALFacilityPythonBinder<I_EventFrameDecoder<RawEventFrameHisto>> bind_decoder_histo(
    [](auto &module, auto &class_binding) {
        class_binding
            .def(
                "add_event_frame_callback",
                +[](I_EventFrameDecoder<RawEventFrameHisto> &self, py::object object) {
                    std::function<void(const RawEventFrameHisto &)> gil_cb = [=](const RawEventFrameHisto &frame) {
                        // Wrap memory space as a python readable buffer.
                        py::gil_scoped_acquire acquire;
                        object(frame);
                    };
                    return self.add_event_frame_callback(gil_cb);
                },
                pybind_doc_hal["Metavision::I_EventFrameDecoder::add_event_frame_callback"])
            .def("decode", &decode_wrapper_histo, py::arg("RawData"),
                 "Decodes raw data. Identifies the events in the buffer and dispatches it to the instance "
                 "Event Decoder corresponding to each event type\n"
                 "\n"
                 "Args:\n"
                 "    RawData: Numpy array of Event Frames\n")
            .def("get_height", &I_EventFrameDecoder<RawEventFrameHisto>::get_height, "get height")
            .def("get_width", &I_EventFrameDecoder<RawEventFrameHisto>::get_width, "get width")
            .def("remove_callback", &I_EventFrameDecoder<RawEventFrameHisto>::remove_callback,
                 pybind_doc_hal["Metavision::I_EventFrameDecoder::remove_callback"]);
    },
    "I_EventFrameDecoder_RawEventFrameHisto");

} // namespace Metavision
