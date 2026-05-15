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

#ifndef METAVISION_UTILS_PYBIND_POD_EVENT_BUFFER_H
#define METAVISION_UTILS_PYBIND_POD_EVENT_BUFFER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace Metavision {

template<typename T>
struct PODEventBuffer {
    PODEventBuffer(size_t size = 0) {
        buffer_.resize(size);
    }

    PODEventBuffer(const PODEventBuffer &other) = delete;

    PODEventBuffer(PODEventBuffer &&other) noexcept {
        buffer_ = std::move(other.buffer_);
    }

    ~PODEventBuffer() = default;

    py::none resize(size_t size) {
        buffer_.resize(size);
        return py::none();
    }

    py::array_t<T> numpy(bool copy = false) {
        if (buffer_.empty() || copy) {
            return py::array_t<T>(buffer_.size(), buffer_.data());
        }

        auto capsule = py::capsule(buffer_.data(), [](void *v) {});
        return py::array_t<T>(buffer_.size(), buffer_.data(), capsule);
    }

    py::buffer_info buffer_info() {
        return py::buffer_info(buffer_.data(),                     // pointer to buffer
                               sizeof(T),                          // size of one element
                               py::format_descriptor<T>::format(), // python struct-style format descriptor
                               1,                                  // number of dimensions
                               {buffer_.size()},                   // shape
                               {sizeof(T)});                       // stride
    }

    std::vector<T> buffer_;
};

template<typename EventType>
void export_PODEventBuffer(py::module &m, const std::string &event_buffer_name) {
    using EventBuffer = PODEventBuffer<EventType>;

    py::class_<EventBuffer, std::shared_ptr<EventBuffer>>(m, event_buffer_name.c_str(), py::buffer_protocol())
        .def_buffer(&EventBuffer::buffer_info)
        .def(py::init<size_t>(), "Constructor", py::arg("size") = 0)
        .def("resize", &EventBuffer::resize, py::arg("size"),
             "resizes the buffer to the specified size\n"
             "\n"
             "   :size: the new size of the buffer")
        .def("numpy", &EventBuffer::numpy, py::arg("copy") = false,
             "Converts to a numpy array\n"
             "\n",
             "   :copy: if True, allocates new memory and returns a copy of the events. If False, use the same memory")
        .def("_buffer_info", &EventBuffer::buffer_info);
}

template<typename T>
PODEventBuffer<T> getEmptyPODBuffer() {
    return PODEventBuffer<T>(0);
}

static const char doc_get_empty_output_buffer_str[] =
    "This function returns an empty buffer of events of the correct type, "
    "which can later on be used as output_buf when calling `process_events()`";

} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_POD_EVENT_BUFFER_H
