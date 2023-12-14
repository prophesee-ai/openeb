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

#ifndef METAVISION_UTILS_PYBIND_ROLLING_EVENT_BUFFER_H
#define METAVISION_UTILS_PYBIND_ROLLING_EVENT_BUFFER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metavision/sdk/base/utils/python_bindings_doc.h"
#include "metavision/sdk/core/utils/rolling_event_buffer.h"

namespace py = pybind11;

namespace Metavision {
static const char doc_insert_numpy_array_str[] = "This function inserts events from a numpy array into the rolling "
                                                 "buffer based on the current mode (N_US or N_EVENTS)\n"
                                                 "    :input_np: input chunk of events\n";

static const char doc_insert_buffer_str[] = "This function inserts events from an event buffer into the rolling buffer "
                                            "based on the current mode (N_US or N_EVENTS)\n"
                                            "    :input_buf: input chunk of events\n";

template<typename T>
void insert_numpy_array(RollingEventBuffer<T> &buffer, const py::array_t<T> &in) {
    auto info = in.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Bad input numpy array");
    }
    auto nelem   = static_cast<size_t>(info.shape[0]);
    auto *in_ptr = static_cast<T *>(info.ptr);

    buffer.insert_events(in_ptr, in_ptr + nelem);
}

template<typename T>
void insert_buffer(RollingEventBuffer<T> &buffer, const PODEventBuffer<T> &in) {
    buffer.insert_events(in.buffer_.cbegin(), in.buffer_.cend());
}

template<typename EventType>
void export_rolling_event_buffer(py::module &m, const std::string &event_buffer_name, const PythonBindingsDoc &doc) {
    using RollingEventBuffer = RollingEventBuffer<EventType>;

    py::class_<RollingEventBuffer, std::shared_ptr<RollingEventBuffer>>(m, event_buffer_name.c_str())
        .def(py::init<const RollingEventBufferConfig &>(),
             doc["Metavision::RollingEventBuffer::RollingEventBuffer(const RollingEventBufferConfig "
                 "&config=RollingEventBufferConfig::make_n_events(5000))"])
        .def("size", &RollingEventBuffer::size, doc["Metavision::RollingEventBuffer::size"])
        .def("capacity", &RollingEventBuffer::capacity, doc["Metavision::RollingEventBuffer::capacity"])
        .def("empty", &RollingEventBuffer::empty, doc["Metavision::RollingEventBuffer::empty"])
        .def("clear", &RollingEventBuffer::clear, doc["Metavision::RollingEventBuffer::clear"])
        .def(
            "__iter__", [](const RollingEventBuffer &b) { return py::make_iterator(b.cbegin(), b.cend()); },
            py::keep_alive<0, 1>())
        .def("insert_events", &insert_numpy_array<EventType>, py::arg("input_np"), doc_insert_numpy_array_str)
        .def("insert_events", &insert_buffer<EventType>, py::arg("input_buf"), doc_insert_buffer_str);
}
} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_ROLLING_EVENT_BUFFER_H