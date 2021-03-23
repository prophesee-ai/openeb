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
#include "metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h"

#include "pb_doc_core.h"

namespace Metavision {

void export_on_demand_frame_generation_algorithm(py::module &m) {
    py::class_<OnDemandFrameGenerationAlgorithm, BaseFrameGenerationAlgorithm>(m, "OnDemandFrameGenerationAlgorithm")
        .def(py::init<int, int, uint32_t, const Metavision::ColorPalette &>(), py::arg("width"), py::arg("height"),
             py::arg("accumulation_time_us") = 10000,
             py::arg("palette")              = BaseFrameGenerationAlgorithm::default_palette(),
             pybind_doc_core["Metavision::OnDemandFrameGenerationAlgorithm::OnDemandFrameGenerationAlgorithm"])
        .def("process_events", &process_events_array_async<OnDemandFrameGenerationAlgorithm, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str)
        .def(
            "generate",
            [](OnDemandFrameGenerationAlgorithm &algo, timestamp ts, py::array &frame) {
                if (!py::isinstance<py::array_t<std::uint8_t>>(frame))
                    throw std::invalid_argument("Incompatible input dtype. Must be np.ubyte.");

                uint32_t height, width, channels;
                algo.get_dimension(height, width, channels);

                cv::Mat img_cv;
                Metavision::py_array_to_cv_mat(frame, img_cv, channels == 3);

                return algo.generate(ts, img_cv, false);
            },
            py::arg("ts"), py::arg("frame"), pybind_doc_core["Metavision::OnDemandFrameGenerationAlgorithm::generate"])
        .def("set_accumulation_time_us", &OnDemandFrameGenerationAlgorithm::set_accumulation_time_us,
             py::arg("accumulation_time_us"),
             pybind_doc_core["Metavision::OnDemandFrameGenerationAlgorithm::set_accumulation_time_us"])
        .def("get_accumulation_time_us", &OnDemandFrameGenerationAlgorithm::get_accumulation_time_us,
             pybind_doc_core["Metavision::OnDemandFrameGenerationAlgorithm::get_accumulation_time_us"]);
}

} // namespace Metavision