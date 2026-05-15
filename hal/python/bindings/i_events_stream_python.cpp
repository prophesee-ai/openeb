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
#include "metavision/hal/facilities/i_events_stream.h"
#include "pb_doc_hal.h"

namespace Metavision {
namespace {
auto get_latest_raw_data_wrapper(I_EventsStream &ies) {
    return ies.get_latest_raw_data();
}
} // namespace

static DeviceFacilityGetter<I_EventsStream> getter("get_i_events_stream");

static HALFacilityPythonBinder<I_EventsStream> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def("start", &I_EventsStream::start, pybind_doc_hal["Metavision::I_EventsStream::start"])
            .def("stop", &I_EventsStream::stop, pybind_doc_hal["Metavision::I_EventsStream::stop"])
            .def("poll_buffer", &I_EventsStream::poll_buffer, pybind_doc_hal["Metavision::I_EventsStream::poll_buffer"])
            .def("wait_next_buffer", &I_EventsStream::wait_next_buffer,
                 pybind_doc_hal["Metavision::I_EventsStream::wait_next_buffer"])
            .def("get_latest_raw_data", &get_latest_raw_data_wrapper,
                 pybind_doc_hal["Metavision::I_EventsStream::get_latest_raw_data()"])
            .def("log_raw_data", &I_EventsStream::log_raw_data, py::arg("f"),
                 pybind_doc_hal["Metavision::I_EventsStream::log_raw_data"])
            .def("stop_log_raw_data", &I_EventsStream::stop_log_raw_data,
                 pybind_doc_hal["Metavision::I_EventsStream::stop_log_raw_data"]);
    },
    "I_EventsStream", pybind_doc_hal["Metavision::I_EventsStream"]);

static HALFacilityPythonBinder<DataTransfer::BufferPtr> bind_buffer_ptr(
    [](auto &module, auto &class_binding) {
        class_binding.def("size", &DataTransfer::BufferPtr::size,
                          pybind_doc_hal["Metavision::DataTransfer::BufferPtr::size"]);
    },
    "RawBuffer", pybind_doc_hal["Metavision::DataTransfer::BufferPtr"]);

} // namespace Metavision
