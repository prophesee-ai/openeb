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
#define MODULE_NAME metavision_sdk_stream_internal
#else
#define MODULE_NAME metavision_sdk_stream
#endif

#include <iostream>
#include <pybind11/pybind11.h>
#if defined(__APPLE__)
#include <pybind11/numpy.h>
#include "metavision/sdk/base/events/event_cd.h"
#endif
#include "metavision/sdk/base/utils/python_bindings_doc.h"
#include "pb_doc_stream.h"

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
#include "python_doc_strings.stream.hpp"
#endif

namespace py = pybind11;

namespace Metavision {

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
PythonBindingsDoc pybind_doc_stream(Metavision::PythonDoc::python_doc_strings_stream);

#else
PythonBindingsDoc pybind_doc_stream;
#endif

void export_camera(py::module &);
void export_camera_stream_slicer(py::module &);
void export_hdf5_event_file_writer(py::module &);
void export_raw_evt2_event_file_writer(py::module &);
void export_synced_cameras_stream_slicer(py::module &);
void export_synced_cameras_system_builder(py::module &m);
void export_synced_cameras_system_factory(py::module &m);
} // namespace Metavision

PYBIND11_MODULE(MODULE_NAME, m) {
    // 1. Import dependencies
    try {
        py::module::import("metavision_sdk_base");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading metavision_sdk_base: " << e.what() << std::endl;
        throw(e);
    }

    try {
        py::module::import("metavision_hal");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading metavision_hal: " << e.what() << std::endl;
        throw(e);
    }

#if defined(__APPLE__)
    PYBIND11_NUMPY_DTYPE(Metavision::Event2d, x, y, p, t);
    PYBIND11_NUMPY_DTYPE(Metavision::EventCD, x, y, p, t);
#endif

    Metavision::export_camera(m);
    Metavision::export_camera_stream_slicer(m);
    Metavision::export_hdf5_event_file_writer(m);
    Metavision::export_raw_evt2_event_file_writer(m);
    Metavision::export_synced_cameras_stream_slicer(m);
    Metavision::export_synced_cameras_system_builder(m);
    Metavision::export_synced_cameras_system_factory(m);
}
