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

#ifndef METAVISION_UTILS_PYBIND_EVENT_BUFFER_VIEW_H
#define METAVISION_UTILS_PYBIND_EVENT_BUFFER_VIEW_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
struct EventBufferView {
    EventBufferView(const std::vector<T> &buffer) : buffer_(buffer) {}

    py::array_t<T> numpy(bool copy = false) {
        if (buffer_.empty() || copy)
            return py::array_t<T>(buffer_.size(), buffer_.data());

        auto capsule = py::capsule(buffer_.data(), [](void *v) {});
        return py::array_t<T>(buffer_.size(), buffer_.data(), capsule);
    }

    const std::vector<T> &buffer_;
};

namespace Metavision {

template<typename EventType>
void export_EventBufferView(py::module &m, const std::string &event_buffer_name) {
    auto def_buffer = [](EventBufferView<EventType> &b) -> py::buffer_info {
        return py::buffer_info(const_cast<EventType *>(b.buffer_.data()),  // pointer to buffer
                               sizeof(EventType),                          // size of one element
                               py::format_descriptor<EventType>::format(), // python struct-style format descriptor
                               1,                                          // number of dimensions
                               {b.buffer_.size()},                         // shape
                               {sizeof(EventType)});                       // stride
    };

    py::class_<EventBufferView<EventType>>(m, event_buffer_name.c_str(), py::buffer_protocol())
        .def_buffer(def_buffer)
        .def("numpy", &EventBufferView<EventType>::numpy, py::arg("copy") = false, "Converts to a numpy array\n", "\n",
             "   :copy: if True, allocates new memory and returns a copy of the events. If False, use the same memory");
}
} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_EVENT_BUFFER_VIEW_H
