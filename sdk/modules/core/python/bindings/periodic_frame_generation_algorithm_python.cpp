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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/pybind/async_algorithm_process_helper.h"
#include "metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h"
#include "metavision/sdk/base/utils/object_pool.h"

#include "pb_doc_core.h"

namespace Metavision {

namespace { // anonymous
auto frame_pool = SharedObjectPool<cv::Mat>();
} // anonymous namespace

void export_periodic_frame_generation_algorithm(py::module &m) {
    py::class_<PeriodicFrameGenerationAlgorithm, BaseFrameGenerationAlgorithm>(m, "PeriodicFrameGenerationAlgorithm")
        .def(py::init<int, int, timestamp, double, ColorPalette>(), py::arg("sensor_width"), py::arg("sensor_height"),
             py::arg("accumulation_time_us") = 10000, py::arg("fps") = 0., py::arg("palette") = ColorPalette::Dark,
             "Inherits BaseFrameGenerationAlgorithm. Algorithm that generates frames from events at a fixed rate "
             "(fps). The reference clock used is the one of the input events\n"
             "\n"
             "Args:\n"
             "    sensor_width (int): Sensor's width (in pixels)\n"
             "    sensor_height (int): Sensor's height (in pixels)\n"
             "    accumulation_time_us (timestamp): Accumulation time (in us) (@ref set_accumulation_time_us)\n"
             "    fps (float): The fps at which to generate the frames. The time reference used is the one from the "
             "input events (@ref set_fps) \n"
             "    palette (ColorPalette): The Prophesee's color palette to use (@ref set_color_palette)\n"
             "@throw std::invalid_argument If the input fps is not positive or if the input accumulation time is not "
             "strictly positive\n")
        .def(
            "set_output_callback",
            [](PeriodicFrameGenerationAlgorithm &algo, const py::object &object) {
                PeriodicFrameGenerationAlgorithm::OutputCb cb = [object](timestamp ts, cv::Mat &mat) {
                    auto frame = frame_pool.acquire();
                    cv::swap(*frame, mat);

                    struct SharedDataWrapper {
                        SharedObjectPool<cv::Mat>::ptr_type frame_;
                    };

                    auto *wrapper   = new SharedDataWrapper();
                    wrapper->frame_ = frame; // copy of the shared ptr
                    // this capsule contains a cv::Mat initialized from the input frame. The reference counter is
                    // incremented so that the frame will have the same lifetime as the py object
                    py::capsule capsule(wrapper, [](void *v) { delete reinterpret_cast<SharedDataWrapper *>(v); });

                    if (frame->channels() == 1) {
                        pybind11::array::StridesContainer strides(
                            {static_cast<py::ssize_t>(frame->step), static_cast<py::ssize_t>(frame->elemSize1())});
                        pybind11::array::ShapeContainer shape({frame->rows, frame->cols});
                        py::array py_array(py::dtype::of<uint8_t>(), shape, strides, frame->data, capsule);
                        object(ts, py_array);
                    } else if (frame->channels() == 3) {
                        pybind11::array::StridesContainer strides({static_cast<py::ssize_t>(frame->step),
                                                                   static_cast<py::ssize_t>(frame->channels()),
                                                                   static_cast<py::ssize_t>(frame->elemSize1())});
                        pybind11::array::ShapeContainer shape({frame->rows, frame->cols, 3});
                        py::array py_array(py::dtype::of<uint8_t>(), shape, strides, frame->data, capsule);
                        object(ts, py_array);
                    } else {
                        throw std::runtime_error("Invalid cv::Mat type. Only CV_8UC1 and CV_8UC3 are supported");
                    }
                };
                algo.set_output_callback(cb);
            },
            "Sets a callback to retrieve the frame")
        .def("process_events", &process_events_array_async<PeriodicFrameGenerationAlgorithm, EventCD>,
             py::arg("events_np"), doc_process_events_array_async_str)
        .def("force_generate", &PeriodicFrameGenerationAlgorithm::force_generate,
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::force_generate"])
        .def("set_fps", &PeriodicFrameGenerationAlgorithm::set_fps, py::arg("fps"),
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::set_fps"])
        .def("get_fps", &PeriodicFrameGenerationAlgorithm::get_fps,
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::get_fps"])
        .def("skip_frames_up_to", &PeriodicFrameGenerationAlgorithm::skip_frames_up_to, py::arg("ts"),
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::skip_frames_up_to"])
        .def("reset", &PeriodicFrameGenerationAlgorithm::reset,
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::reset"])
        .def("set_accumulation_time_us", &PeriodicFrameGenerationAlgorithm::set_accumulation_time_us,
             py::arg("accumulation_time_us"),
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::set_accumulation_time_us"])
        .def("get_accumulation_time_us", &PeriodicFrameGenerationAlgorithm::get_accumulation_time_us,
             pybind_doc_core["Metavision::PeriodicFrameGenerationAlgorithm::get_accumulation_time_us"]);
}

} // namespace Metavision