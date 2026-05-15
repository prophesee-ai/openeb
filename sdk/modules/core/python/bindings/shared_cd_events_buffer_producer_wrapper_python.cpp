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
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "metavision/sdk/base/events/event_cd.h"
#include "shared_cd_events_buffer_producer_wrapper.h"
#include "metavision/utils/pybind/async_algorithm_process_helper.h"
#include "pb_doc_core.h"

namespace py = pybind11;

namespace Metavision {

std::function<void(const EventCD *, const EventCD *)>
    SharedCdEventsBufferProducerWrapper::get_process_events_callback() {
    return [this](const EventCD *begin, const EventCD *end) { process_events(begin, end); };
}
struct Memo {
    EventsBufferPtr ptr;
};

static BufferCallback python_callback_wrapper(py::object object) {
    BufferCallback gil_cb = [object](timestamp end_ts, const EventsBufferPtr &buffer) {
        auto memo = new Memo();
        memo->ptr = buffer; // copy of the shared ptr
        // this capsule contains a copy of the event buffers shared pointer, that will be destroyed
        // when python release the memory
        py::capsule capsule = py::capsule(memo, [](void *v) { delete reinterpret_cast<Memo *>(v); });
        // we reinterpret the vector as a numpy array
        auto py_array = py::array_t<Metavision::EventCD>(buffer->size(), buffer->data(), capsule);

        py::gil_scoped_acquire acquire;
        // actual python call
        object(end_ts, py_array);
    };
    return gil_cb;
}

void export_shared_cd_events_buffer_producer(py::module &m) {
    py::class_<SharedCdEventsBufferProducerWrapper>(
        m, "SharedCdEventsBufferProducer",
        "This class splits incoming events into buffers either by number of events or by time slice (in us)\n\n"
        "Incoming events are put in buffer contained in a pool of shared vectors until the buffer is complete"
        "Then a python callback specified by the user is called on the buffer of events.")
        .def(py::init([](py::object object, uint32_t buffers_events_count, uint32_t buffers_time_slice_us,
                         uint32_t buffers_pool_size, uint32_t buffers_preallocation_size) {
                 BufferCallback gil_cb = python_callback_wrapper(object);

                 SharedEventsBufferProducerParameters params;
                 params.buffers_pool_size_          = buffers_pool_size;
                 params.buffers_preallocation_size_ = buffers_preallocation_size;
                 params.buffers_time_slice_us_      = buffers_time_slice_us;
                 params.buffers_events_count_       = buffers_events_count;
                 params.bounded_memory_pool_        = false;
                 return new SharedCdEventsBufferProducerWrapper(params, gil_cb);
             }),
             py::arg("callback"), py::arg("event_count") = 0, py::arg("time_slice_us") = 10000,
             py::arg("buffers_pool_size") = 64, py::arg("buffers_preallocation_size") = 0,
             "Args:\n"
             "    callback (function): python callback taking as input an int coding the last timestamp of the buffer\n"
             "        and a numpy buffer of EventCD.\n"
             "    event_count (int): number of events in each buffer\n"
             "    time_slice_us (int): duration of the buffer in us.\n"
             "    buffers_pool_size (int): Number of shared pointers available in the pool at start. They will be \n"
             "         increased if necessary automatically. Can be left as is in most cases.\n"
             "    buffers_preallocation_size (int): initialization size of vectors in the memory pool. Here again,\n"
             "        this can be left alone in most uses.\n")
        .def("process_events", &process_events_array_async<SharedCdEventsBufferProducerWrapper, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str)
        .def("set_processing_n_us", &SharedCdEventsBufferProducerWrapper::set_processing_n_us,
             pybind_doc_core["Metavision::AsyncAlgorithm::set_processing_n_us"], py::arg("delta_ts"))
        .def("get_processing_n_us", &SharedCdEventsBufferProducerWrapper::get_processing_n_us,
             pybind_doc_core["Metavision::AsyncAlgorithm::get_processing_n_us"])
        .def("set_processing_n_events", &SharedCdEventsBufferProducerWrapper::set_processing_n_events,
             pybind_doc_core["Metavision::AsyncAlgorithm::set_processing_n_events"], py::arg("delta_n_events"))
        .def("get_processing_n_events", &SharedCdEventsBufferProducerWrapper::get_processing_n_events,
             pybind_doc_core["Metavision::AsyncAlgorithm::get_processing_n_events"])
        .def("set_processing_mixed", &SharedCdEventsBufferProducerWrapper::set_processing_mixed,
             pybind_doc_core["Metavision::AsyncAlgorithm::set_processing_mixed"], py::arg("delta_n_events"),
             py::arg("delta_ts"))
        .def("flush", &SharedCdEventsBufferProducerWrapper::flush,
             "Flushes the last buffers when the file is done, producing a last incomplete buffer.")

        .def("reset", &SharedCdEventsBufferProducerWrapper::reset, "Resets the buffer.")
        .def("get_process_events_callback", &SharedCdEventsBufferProducerWrapper::get_process_events_callback,
             "Returns a callback to be passed to the event_cd decoder from Metavision HAL.");
}

} // namespace Metavision