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
#define MODULE_NAME metavision_sdk_ui_internal
#else
#define MODULE_NAME metavision_sdk_ui
#endif

#include <iostream>

#include <pybind11/pybind11.h>

#include "pb_doc_ui.h"

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
#include "python_doc_strings.ui.hpp"
#endif

namespace py = pybind11;

namespace Metavision {

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
PythonBindingsDoc pybind_doc_ui(Metavision::PythonDoc::python_doc_strings_ui);
#else
PythonBindingsDoc pybind_doc_ui;
#endif

void export_base_window(py::module &);
void export_event_loop(py::module &);
void export_mt_window(py::module &);
void export_ui_events(py::module &);
void export_window(py::module &);

} // namespace Metavision

PYBIND11_MODULE(MODULE_NAME, m) {
    // 1. Import dependencies
    try {
        py::module::import("metavision_sdk_base");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading metavision_sdk_base: " << e.what() << std::endl;
        throw(e);
    }

    // 2. Export event types
    Metavision::export_ui_events(m);

    // 3. Export algos
    Metavision::export_base_window(m);
    Metavision::export_event_loop(m);
    Metavision::export_mt_window(m);
    Metavision::export_window(m);
}
