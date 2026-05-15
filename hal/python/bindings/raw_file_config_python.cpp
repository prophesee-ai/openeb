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

#include "hal_python_binder.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "pb_doc_hal.h"

namespace Metavision {

static HALClassPythonBinder<RawFileConfig> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def(py::init<>())
            .def(py::init<const RawFileConfig &>())
            .def_readwrite("n_events_to_read", &RawFileConfig::n_events_to_read_,
                           pybind_doc_hal["Metavision::RawFileConfig::n_events_to_read_"])
            .def_readwrite("n_read_buffers", &RawFileConfig::n_read_buffers_,
                           pybind_doc_hal["Metavision::RawFileConfig::n_read_buffers_"])
            .def_readwrite("do_time_shifting", &RawFileConfig::do_time_shifting_,
                           pybind_doc_hal["Metavision::RawFileConfig::do_time_shifting_"])
            .def_readwrite("build_index", &RawFileConfig::build_index_,
                           pybind_doc_hal["Metavision::RawFileConfig::build_index_"]);
    },
    "RawFileConfig", pybind_doc_hal["Metavision::RawFileConfig"]);

} // namespace Metavision
