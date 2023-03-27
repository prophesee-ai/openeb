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

#include "metavision/sdk/core/utils/raw_event_frame_converter.h"

#include "pb_doc_core.h"

namespace py = pybind11;

namespace Metavision {

RawEventFrameConverter *create_RawEventFrameConverter(unsigned height, unsigned width, bool use_CHW = false) {
    HistogramFormat format = HistogramFormat::HWC;
    if (use_CHW) {
        format = HistogramFormat::CHW;
    }
    return new RawEventFrameConverter(height, width, 2, format);
}

void RawEventFrameConverter_set_HWC_helper(RawEventFrameConverter &frame_converter) {
    frame_converter.set_format(HistogramFormat::HWC);
}

void RawEventFrameConverter_set_CHW_helper(RawEventFrameConverter &frame_converter) {
    frame_converter.set_format(HistogramFormat::CHW);
}

py::array_t<int8_t> RawEventFrameConverter_convert_diff_to_int8(const RawEventFrameConverter &frame_converter,
                                                                const RawEventFrameDiff &d) {
    std::unique_ptr<EventFrameDiff<int8_t>> frame = frame_converter.convert<int8_t>(d);
    assert(frame->get_size() == frame_converter.get_height() * frame_converter.get_width());
    std::vector<py::ssize_t> shape = {frame_converter.get_height(), frame_converter.get_width()};
    return py::array_t<int8_t>(shape, frame->get_data().data());
}

py::array_t<uint8_t> RawEventFrameConverter_convert_histo_to_uint8(const RawEventFrameConverter &frame_converter,
                                                                   const RawEventFrameHisto &h) {
    std::unique_ptr<EventFrameHisto<uint8_t>> frame = frame_converter.convert<uint8_t>(h);
    assert(frame->get_size() == 2 * frame_converter.get_height() * frame_converter.get_width());
    std::vector<py::ssize_t> shape;
    if (frame_converter.get_format() == HistogramFormat::HWC) {
        shape = {frame_converter.get_height(), frame_converter.get_width(), 2};
    } else {
        shape = {2, frame_converter.get_height(), frame_converter.get_width()};
    }
    return py::array_t<uint8_t>(shape, frame->get_data().data());
}

void export_raw_event_frame_converter(py::module &m) {
    py::class_<RawEventFrameConverter>(m, "RawEventFrameConverter",
                                       pybind_doc_core["Metavision::RawEventFrameConverter"])
        .def(py::init(&create_RawEventFrameConverter), py::arg("height"), py::arg("width"), py::arg("use_CHW") = false,
             "Creates a RawEventFrameConverter")
        .def("set_HWC", &RawEventFrameConverter_set_HWC_helper, "Set histo output format to HWC")
        .def("set_CHW", &RawEventFrameConverter_set_CHW_helper, "Set histo output format to CHW")
        .def("convert_diff", &RawEventFrameConverter_convert_diff_to_int8,
             "Converts a RawEventFrameDiff into a proper diff frame")
        .def("convert_histo", &RawEventFrameConverter_convert_histo_to_uint8,
             "Converts a RawEventFrameHisto into a proper histo frame");
}

} // namespace Metavision
