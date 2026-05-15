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
#ifdef _WIN32
#define MODULE_NAME metavision_sdk_base_internal
#else
#define MODULE_NAME metavision_sdk_base
#endif

#include <iostream>
#include <pybind11/pybind11.h>

#include "metavision/sdk/base/utils/python_bindings_doc.h"
#include "pb_doc_base.h"

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
#include "python_doc_strings.base.hpp"
#endif

namespace py = pybind11;

namespace Metavision {

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
PythonBindingsDoc pybind_doc_base(Metavision::PythonDoc::python_doc_strings_base);

#else
PythonBindingsDoc pybind_doc_base;
#endif

void export_event_cd(py::module &m);
void export_event_ext_trigger(py::module &m);
void export_raw_event_frame_diff_and_histo(py::module &m);
void export_software_info(py::module &m);
void export_debug_buffer_info(py::module &m);
void export_generic_header(py::module &m);
} // namespace Metavision

PYBIND11_MODULE(MODULE_NAME, m) {
    try {
        py::module::import("numpy");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading numpy: " << e.what() << std::endl;
        throw(e);
    }

    // Export events
    Metavision::export_event_cd(m);
    Metavision::export_event_ext_trigger(m);
    Metavision::export_raw_event_frame_diff_and_histo(m);

    // Export tools
    Metavision::export_software_info(m);
    Metavision::export_debug_buffer_info(m);
    Metavision::export_generic_header(m);
}
