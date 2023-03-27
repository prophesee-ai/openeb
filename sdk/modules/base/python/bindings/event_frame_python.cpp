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

#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/base/events/raw_event_frame_histo.h"

namespace py = pybind11;

namespace Metavision {

py::array_t<int8_t> RawEventFrameDiff_to_numpy(RawEventFrameDiff &frame) {
    assert(frame.buffer_size() == frame.get_config().height * frame.get_config().width);
    std::vector<py::ssize_t> shape = {frame.get_config().height, frame.get_config().width};
    return py::array_t<int8_t>(shape, frame.get_data().data());
}

void export_raw_event_frame_diff_and_histo(py::module &m) {
    py::class_<RawEventFrameDiff>(m, "RawEventFrameDiff")
        .def(py::init<unsigned, unsigned>(), py::arg("height"), py::arg("width"))
        .def("buffer_size", &RawEventFrameDiff::buffer_size)
        .def("numpy", &RawEventFrameDiff_to_numpy, "Converts to a numpy array");

    py::class_<RawEventFrameHisto>(m, "RawEventFrameHisto")
        .def(py::init<unsigned, unsigned>(), py::arg("height"), py::arg("width"))
        .def("buffer_size", &RawEventFrameHisto::buffer_size);
}

} // namespace Metavision
