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

#include <pybind11/stl.h>

#include "metavision/utils/pybind/py_array_to_cv_mat.h"
#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"
#include "metavision/sdk/core/utils/rolling_event_buffer.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "pb_doc_core.h"

namespace Metavision {

namespace { // anonymous

void generate_frame_from_event_array(const py::array_t<EventCD> &events, py::array &frame,
                                     uint32_t accumulation_time_us, const Metavision::ColorPalette &palette) {
    cv::Mat img_cv;
    Metavision::py_array_to_cv_mat(frame, img_cv, true);

    auto info = events.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Bad input numpy array dimension " + std::to_string(info.ndim) +
                                 " should be equal to 1");
    }
    auto nelem   = static_cast<size_t>(info.shape[0]);
    auto *in_ptr = static_cast<EventCD *>(info.ptr);
    BaseFrameGenerationAlgorithm::generate_frame_from_events(in_ptr, in_ptr + nelem, img_cv, accumulation_time_us,
                                                             palette);
}

void generate_frame_from_event_rolling_buffer(const RollingEventBuffer<EventCD> &events, py::array &frame,
                                              uint32_t accumulation_time_us, const Metavision::ColorPalette &palette) {
    cv::Mat img_cv;
    Metavision::py_array_to_cv_mat(frame, img_cv, true);

    BaseFrameGenerationAlgorithm::generate_frame_from_events(events.cbegin(), events.cend(), img_cv,
                                                             accumulation_time_us, palette);
}

py::tuple Vec3b_to_tuple(const cv::Vec3b &color) {
    return py::make_tuple(color[0], color[1], color[2]);
}

void set_colors_helper(BaseFrameGenerationAlgorithm &algo, const std::vector<uint8_t> &background_color,
                       const std::vector<uint8_t> &on_color, const std::vector<uint8_t> &off_color, bool colored) {
    cv::Vec3b bg, on, off;
    if ((colored == true) && ((background_color.size() != 3) || (on_color.size() != 3) || (off_color.size() != 3))) {
        std::ostringstream oss;
        oss << "Expecting color image, but colors are not in RGB format" << std::endl;
        oss << "  background_color tuple size (expected 3): " << background_color.size() << std::endl;
        oss << "  on_color tuple size (expected 3): " << on_color.size() << std::endl;
        oss << "  off_color tuple size (expected 3): " << off_color.size() << std::endl;
        throw std::runtime_error(oss.str());
    }

    if ((colored == false) && ((background_color.size() != 1) || (on_color.size() != 1) || (off_color.size() != 1))) {
        std::ostringstream oss;
        oss << "Expecting grayscale image, but colors are not in RGB format" << std::endl;
        oss << "  background_color tuple size (expected 1): " << background_color.size() << std::endl;
        oss << "  on_color tuple size (expected 1): " << on_color.size() << std::endl;
        oss << "  off_color tuple size (expected 1): " << off_color.size() << std::endl;
        throw std::runtime_error(oss.str());
    }

    if (colored) {
        bg[0]  = background_color[0];
        bg[1]  = background_color[1];
        bg[2]  = background_color[2];
        on[0]  = on_color[0];
        on[1]  = on_color[1];
        on[2]  = on_color[2];
        off[0] = off_color[0];
        off[1] = off_color[1];
        off[2] = off_color[2];
    } else {
        bg[0]  = background_color[0];
        on[0]  = on_color[0];
        off[0] = off_color[0];
    }
    algo.set_colors(bg, on, off, colored);
}

} // anonymous namespace

void export_base_frame_generation_algorithm(py::module &m) {
    static constexpr char generate_doc_str[] =
        "Stand-alone (static) method to generate a frame from events\n"
        "\n"
        "   All events in the interval ]t - dt, t] are used where t the timestamp of the last event in "
        "the buffer, and dt is accumulation_time_us. If accumulation_time_us is kept to 0, all input events are "
        "used.\n"
        "   If there is no events, a frame filled with the background color will be generated\n"
        "\n"
        "   :events: Numpy structured array whose fields are ('x', 'y', 'p', 't'). Note that this order is "
        "mandatory\n"
        "   :frame: Pre-allocated frame that will be filled with CD events. It must have the same geometry as the "
        "input"
        " event source, and the color corresponding to the given palette (3 channels by default)\n"
        "   :accumulation_time_us: Time range of events to update the frame with (in us). 0 to use all events.\n"
        "   :palette: The Prophesee's color palette to use";

    py::class_<BaseFrameGenerationAlgorithm>(m, "BaseFrameGenerationAlgorithm")
        .def_static("generate_frame", &generate_frame_from_event_array, py::arg("events"), py::arg("frame"),
                    py::arg("accumulation_time_us") = 0,
                    py::arg("palette")              = BaseFrameGenerationAlgorithm::default_palette(), generate_doc_str)
        .def_static("generate_frame", &generate_frame_from_event_rolling_buffer, py::arg("events"), py::arg("frame"),
                    py::arg("accumulation_time_us") = 0,
                    py::arg("palette")              = BaseFrameGenerationAlgorithm::default_palette(), generate_doc_str)
        .def_static(
            "bg_color_default", []() { return Vec3b_to_tuple(BaseFrameGenerationAlgorithm::bg_color_default()); },
            pybind_doc_core["Metavision::BaseFrameGenerationAlgorithm::bg_color_default"])
        .def_static(
            "on_color_default", []() { return Vec3b_to_tuple(BaseFrameGenerationAlgorithm::on_color_default()); },
            pybind_doc_core["Metavision::BaseFrameGenerationAlgorithm::on_color_default"])
        .def_static(
            "off_color_default", []() { return Vec3b_to_tuple(BaseFrameGenerationAlgorithm::off_color_default()); },
            pybind_doc_core["Metavision::BaseFrameGenerationAlgorithm::off_color_default"])
        .def("set_colors", &set_colors_helper, py::arg("background_color"), py::arg("on_color"), py::arg("off_color"),
             py::arg("colored") = true, pybind_doc_core["Metavision::BaseFrameGenerationAlgorithm::set_colors"])
        .def("set_color_palette", &BaseFrameGenerationAlgorithm::set_color_palette, py::arg("palette"),
             pybind_doc_core["Metavision::BaseFrameGenerationAlgorithm::set_color_palette"])
        .def(
            "get_dimension",
            [](const BaseFrameGenerationAlgorithm &algo) {
                uint32_t height, width, channels;
                algo.get_dimension(height, width, channels);
                return py::make_tuple(height, width, channels);
            },
            "Gets the frame's dimension, a tuple (height, width, channels)");
}

} // namespace Metavision
