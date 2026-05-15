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

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/pybind/async_algorithm_process_helper.h"
#include "metavision/utils/pybind/py_array_to_cv_mat.h"
#include "metavision/sdk/core/algorithms/contrast_map_generation_algorithm.h"

#include "pb_doc_core.h"

namespace Metavision {

void export_contrast_map_generation_algorithm(py::module &m) {
    py::class_<ContrastMapGenerationAlgorithm>(m, "ContrastMapGenerationAlgorithm")
        .def(py::init<unsigned int, unsigned int, float, float>(), py::arg("width"), py::arg("height"),
             py::arg("contrast_on") = 1.2f, py::arg("contrast_off") = -1.f,
             pybind_doc_core["Metavision::ContrastMapGenerationAlgorithm::ContrastMapGenerationAlgorithm"])
        .def("process_events", &process_events_array_async<ContrastMapGenerationAlgorithm, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str)
        .def(
            "generate",
            [](ContrastMapGenerationAlgorithm &algo, py::array &frame) {
                cv::Mat_<float> frame_cv = to_cv_mat_<float>(frame);
                algo.generate(frame_cv);
            },
            py::arg("frame"),
            pybind_doc_core["Metavision::ContrastMapGenerationAlgorithm::generate(cv::Mat_< float > &contrast_map)"])
        .def(
            "generate",
            [](ContrastMapGenerationAlgorithm &algo, py::array &frame, float tonemapping_factor,
               float tonemapping_bias) {
                cv::Mat_<uchar> frame_cv = to_cv_mat_<uchar>(frame);
                algo.generate(frame_cv, tonemapping_factor, tonemapping_bias);
            },
            py::arg("frame"), py::arg("tonemapping_factor"), py::arg("tonemapping_bias"),
            pybind_doc_core["Metavision::ContrastMapGenerationAlgorithm::generate(cv::Mat_< uchar > "
                            "&contrast_map_tonnemapped, float tonemapping_factor, float tonemapping_bias)"])
        .def("reset", &ContrastMapGenerationAlgorithm::reset,
             pybind_doc_core["Metavision::ContrastMapGenerationAlgorithm::reset"]);
}

} // namespace Metavision
