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
#define MODULE_NAME metavision_sdk_core_internal
#else
#define MODULE_NAME metavision_sdk_core
#endif

#include <iostream>
#include <pybind11/pybind11.h>
#if defined(__APPLE__)
#include <pybind11/numpy.h>
#include "metavision/sdk/base/events/event_cd.h"
#endif
#include "metavision/sdk/base/utils/python_bindings_doc.h"
#include "pb_doc_core.h"

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
#include "python_doc_strings.core.hpp"
#endif

namespace py = pybind11;

namespace Metavision {

#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
PythonBindingsDoc pybind_doc_core(Metavision::PythonDoc::python_doc_strings_core);

#else
PythonBindingsDoc pybind_doc_core;
#endif

void export_event_bbox(py::module &);
void export_event_tracked_box(py::module &);
void export_base_frame_generation_algorithm(py::module &);
void export_contrast_map_generation_algorithm(py::module &);
void export_event_preprocessor(py::module &);
void export_event_rescaler_algorithm(py::module &m);
void export_events_integration_algorithm(py::module &);
void export_colors(py::module &);
void export_adaptive_rate_events_splitter_algorithm(py::module &);
void export_flip_x_algorithm(py::module &);
void export_flip_y_algorithm(py::module &);
void export_on_demand_frame_generation_algorithm(py::module &);
void export_mostrecent_timestamp_buffer(py::module &m);
void export_periodic_frame_generation_algorithm(py::module &);
void export_polarity_filter_algorithm(py::module &);
void export_polarity_inverter_algorithm(py::module &);
void export_raw_event_frame_converter(py::module &);
void export_roi_filter_algorithm(py::module &);
void export_roi_mask_algorithm(py::module &);
void export_rotate_events_algorithm(py::module &);
void export_shared_cd_events_buffer_producer(py::module &);
void export_stream_logger_algorithm(py::module &);
void export_transpose_events_algorithm(py::module &);
void export_rolling_event_cd_buffer(py::module &);
} // namespace Metavision

PYBIND11_MODULE(MODULE_NAME, m) {
    // 1. Import dependencies
    try {
        py::module::import("metavision_sdk_base");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading metavision_sdk_base: " << e.what() << std::endl;
        throw(e);
    }

#if defined(__APPLE__)
    PYBIND11_NUMPY_DTYPE(Metavision::Event2d, x, y, p, t);
    PYBIND11_NUMPY_DTYPE(Metavision::EventCD, x, y, p, t);
#endif

    // 2. Export event types
    Metavision::export_event_bbox(m);
    Metavision::export_event_tracked_box(m);
    Metavision::export_colors(m);
    Metavision::export_mostrecent_timestamp_buffer(m);

    // 3. Export algos
    Metavision::export_adaptive_rate_events_splitter_algorithm(m);
    Metavision::export_base_frame_generation_algorithm(m);
    Metavision::export_contrast_map_generation_algorithm(m);
    Metavision::export_event_preprocessor(m);
    Metavision::export_event_rescaler_algorithm(m);
    Metavision::export_events_integration_algorithm(m);
    Metavision::export_flip_x_algorithm(m);
    Metavision::export_flip_y_algorithm(m);
    Metavision::export_on_demand_frame_generation_algorithm(m);
    Metavision::export_periodic_frame_generation_algorithm(m);
    Metavision::export_polarity_filter_algorithm(m);
    Metavision::export_polarity_inverter_algorithm(m);
    Metavision::export_raw_event_frame_converter(m);
    Metavision::export_roi_filter_algorithm(m);
    Metavision::export_roi_mask_algorithm(m);
    Metavision::export_rotate_events_algorithm(m);
    Metavision::export_shared_cd_events_buffer_producer(m);
    Metavision::export_stream_logger_algorithm(m);
    Metavision::export_transpose_events_algorithm(m);

    // 4. Export utils
    Metavision::export_rolling_event_cd_buffer(m);
}
