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
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "pb_doc_hal.h"

// Needed to avoid copies of vectors of RawData
PYBIND11_MAKE_OPAQUE(Metavision::DataTransfer::Buffer);

namespace Metavision {

namespace {
    std::shared_ptr<DataTransfer::Buffer> get_latest_raw_data_wrapper(I_EventsStream *ies) {
        auto data = ies->get_latest_raw_data();
        // If we return nullptr, python doesn't know the type of the objects and considers it "None".
        // This causes issues when trying to call decode on an empty buffer which has been possible so far.
        // So return a shared pointer to an empty vector not to break user scripts
        if (!data) {
            return std::make_shared<DataTransfer::Buffer>();
        }
        return data;
    }
}

static DeviceFacilityGetter<I_EventsStream> getter("get_i_events_stream");

static HALFacilityPythonBinder<I_EventsStream> bind(
    [](auto &module, auto &class_binding) {

        py::bind_vector<DataTransfer::Buffer, DataTransfer::BufferPtr>(module, "RawDataBuffer");
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

} // namespace Metavision
