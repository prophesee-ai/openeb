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
#include "metavision/sdk/core/algorithms/events_integration_algorithm.h"

#include "pb_doc_core.h"

namespace Metavision {

void export_events_integration_algorithm(py::module &m) {
    py::class_<EventsIntegrationAlgorithm>(m, "EventsIntegrationAlgorithm")
        .def(py::init<unsigned int, unsigned int, Metavision::timestamp, float, float, int, int, float>(),
             py::arg("width"), py::arg("height"), py::arg("decay_time") = 1'000'000, py::arg("contrast_on") = 1.2f,
             py::arg("contrast_off") = -1.f, py::arg("tonemapping_max_ev_count") = 5,
             py::arg("gaussian_blur_kernel_radius") = 1, py::arg("diffusion_weight") = 0.f,
             pybind_doc_core["Metavision::EventsIntegrationAlgorithm::EventsIntegrationAlgorithm"])
        .def("process_events", &process_events_array_async<EventsIntegrationAlgorithm, EventCD>, py::arg("events_np"),
             doc_process_events_array_async_str)
        .def(
            "generate",
            [](EventsIntegrationAlgorithm &algo, py::array &frame) {
                if (!py::isinstance<py::array_t<std::uint8_t>>(frame))
                    throw std::invalid_argument("Incompatible input dtype. Must be np.ubyte.");

                cv::Mat img_cv;
                Metavision::py_array_to_cv_mat(frame, img_cv, false);

                return algo.generate(img_cv);
            },
            py::arg("frame"), pybind_doc_core["Metavision::EventsIntegrationAlgorithm::generate"])
        .def("reset", &EventsIntegrationAlgorithm::reset,
             pybind_doc_core["Metavision::EventsIntegrationAlgorithm::reset"]);
}

} // namespace Metavision
