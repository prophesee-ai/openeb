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
#include "metavision/sdk/core/preprocessors/event_preprocessor.h"
#include "metavision/sdk/core/preprocessors/diff_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_diff_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_histo_processor.h"
#include "metavision/sdk/core/preprocessors/histo_processor.h"
#include "metavision/sdk/core/preprocessors/event_cube_processor.h"
#include "metavision/sdk/core/preprocessors/time_surface_processor.h"

#include "metavision/utils/pybind/pod_event_buffer.h"
#include "metavision/utils/pybind/sync_algorithm_process_helper.h"

#include "pb_doc_core.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace {

using ArrayIt                = typename Metavision::EventCD *;
using EventPreprocessorArray = Metavision::EventPreprocessor<ArrayIt>;

bool are_same_type(const py::buffer_info &b, Metavision::BaseType t) {
    switch (t) {
    case Metavision::BaseType::BOOL:
        return py::detail::compare_buffer_info<bool>::compare(b);
    case Metavision::BaseType::UINT8:
        return py::detail::compare_buffer_info<std::uint8_t>::compare(b);
    case Metavision::BaseType::UINT16:
        return py::detail::compare_buffer_info<std::uint16_t>::compare(b);
    case Metavision::BaseType::UINT32:
        return py::detail::compare_buffer_info<std::uint32_t>::compare(b);
    case Metavision::BaseType::UINT64:
        return py::detail::compare_buffer_info<std::uint64_t>::compare(b);
    case Metavision::BaseType::INT8:
        return py::detail::compare_buffer_info<std::int8_t>::compare(b);
    case Metavision::BaseType::INT16:
        return py::detail::compare_buffer_info<std::int16_t>::compare(b);
    case Metavision::BaseType::INT32:
        return py::detail::compare_buffer_info<std::int32_t>::compare(b);
    case Metavision::BaseType::INT64:
        return py::detail::compare_buffer_info<std::int64_t>::compare(b);
    case Metavision::BaseType::FLOAT32:
        return py::detail::compare_buffer_info<float>::compare(b);
    case Metavision::BaseType::FLOAT64:
        return py::detail::compare_buffer_info<double>::compare(b);
    default:
        throw std::runtime_error("No comparison available for type " + Metavision::to_string(t));
    }
};

std::string compute_descriptor(Metavision::BaseType t) {
    switch (t) {
    case Metavision::BaseType::BOOL:
        return py::format_descriptor<bool>::format();
    case Metavision::BaseType::UINT8:
        return py::format_descriptor<std::uint8_t>::format();
    case Metavision::BaseType::UINT16:
        return py::format_descriptor<std::uint16_t>::format();
    case Metavision::BaseType::UINT32:
        return py::format_descriptor<std::uint32_t>::format();
    case Metavision::BaseType::UINT64:
        return py::format_descriptor<std::uint64_t>::format();
    case Metavision::BaseType::INT8:
        return py::format_descriptor<std::int8_t>::format();
    case Metavision::BaseType::INT16:
        return py::format_descriptor<std::int16_t>::format();
    case Metavision::BaseType::INT32:
        return py::format_descriptor<std::int32_t>::format();
    case Metavision::BaseType::INT64:
        return py::format_descriptor<std::int64_t>::format();
    case Metavision::BaseType::FLOAT32:
        return py::format_descriptor<float>::format();
    case Metavision::BaseType::FLOAT64:
        return py::format_descriptor<double>::format();
    default:
        throw std::runtime_error("No description available for type " + Metavision::to_string(t));
    }
}

void EventPreprocessorArray_check_output_frame_validity(const EventPreprocessorArray &cdproc, py::array &frame_np) {
    const auto info_frame  = frame_np.request();
    const auto &shape      = cdproc.get_output_shape();
    const auto &dimensions = shape.dimensions;
    const size_t nb_dim    = dimensions.size();
    if (info_frame.ndim != static_cast<int>(nb_dim)) {
        std::ostringstream oss;
        oss << "Frame dim should be " << nb_dim << ". frame_np.ndim : " << info_frame.ndim << std::endl;
        throw std::runtime_error(oss.str());
    }
    const auto type_descr = compute_descriptor(cdproc.get_output_type());
    if (!are_same_type(info_frame, cdproc.get_output_type())) {
        std::ostringstream oss;
        oss << "Frame type don't match ! Got py array " << info_frame.format << " but expected " << type_descr
            << std::endl;
        throw std::runtime_error(oss.str());
    }
    if (static_cast<size_t>(info_frame.size) != shape.get_nb_values()) {
        std::ostringstream oss;
        oss << "Frame size should be: " << shape.get_nb_values();
        oss << ". frame_np.size: " << info_frame.size << std::endl;
        throw std::runtime_error(oss.str());
    }
    for (unsigned int i = 0; i < nb_dim; ++i) {
        if (dimensions[i].dim != static_cast<int>(info_frame.shape[i])) {
            std::stringstream msg;
            msg << "Incompatible dimension " << i << ". Expected " << dimensions[i].dim << " but got "
                << info_frame.shape[i] << std::endl;
            throw std::runtime_error(msg.str());
        }
    }
}

void EventPreprocessorArray_process_events_array(const EventPreprocessorArray &cdproc,
                                                 const Metavision::timestamp cur_frame_start_ts,
                                                 const py::array_t<Metavision::EventCD> &events, py::array &frame_np) {
    const auto info_events = events.request();
    if (info_events.ndim != 1) {
        throw std::runtime_error("Wrong events dim");
    }
    const auto events_ptr = static_cast<Metavision::EventCD *>(info_events.ptr);
    const int nb_events   = info_events.shape[0];

    EventPreprocessorArray_check_output_frame_validity(cdproc, frame_np);
    auto info_frame = frame_np.request();

    Metavision::Tensor t(cdproc.get_output_shape(), cdproc.get_output_type(),
                         reinterpret_cast<std::byte *>(info_frame.ptr), false);
    cdproc.process_events(cur_frame_start_ts, events_ptr, events_ptr + nb_events, t);
}

template<typename T>
py::array_t<T> produce_array(const Metavision::TensorShape &shape, const Metavision::BaseType type) {
    const auto &dimensions = shape.dimensions;
    auto frame_np          = py::array_t<T>({dimensions[0].dim, dimensions[1].dim, dimensions[2].dim});
    memset(frame_np.request().ptr, 0, shape.get_nb_values() * sizeof(T));
    return frame_np;
}

py::array EventPreprocessorArray_init_output_tensor(const EventPreprocessorArray &cdproc) {
    const auto &shape = cdproc.get_output_shape();
    const auto &type  = cdproc.get_output_type();
    switch (type) {
    case Metavision::BaseType::BOOL:
        return produce_array<bool>(shape, type);
    case Metavision::BaseType::UINT8:
        return produce_array<std::uint8_t>(shape, type);
    case Metavision::BaseType::UINT16:
        return produce_array<std::uint16_t>(shape, type);
    case Metavision::BaseType::UINT32:
        return produce_array<std::uint32_t>(shape, type);
    case Metavision::BaseType::UINT64:
        return produce_array<std::uint64_t>(shape, type);
    case Metavision::BaseType::INT8:
        return produce_array<std::int8_t>(shape, type);
    case Metavision::BaseType::INT16:
        return produce_array<std::int16_t>(shape, type);
    case Metavision::BaseType::INT32:
        return produce_array<std::int32_t>(shape, type);
    case Metavision::BaseType::INT64:
        return produce_array<std::int64_t>(shape, type);
    case Metavision::BaseType::FLOAT32:
        return produce_array<float>(shape, type);
    case Metavision::BaseType::FLOAT64:
        return produce_array<double>(shape, type);
    default:
        throw std::runtime_error("No implementation for type " + Metavision::to_string(type));
    };
}

// End Helpers for EventPreprocessor

// Helpers factory
EventPreprocessorArray *create_EventPreprocessorArrayDiff(int event_input_width, int event_input_height,
                                                          float max_incr_per_pixel,
                                                          float clip_value_after_normalization, float width_scale = 1.f,
                                                          float height_scale = 1.f) {
    return new Metavision::DiffProcessor<ArrayIt>(event_input_width, event_input_height, max_incr_per_pixel,
                                                  clip_value_after_normalization, width_scale, height_scale);
}

EventPreprocessorArray *create_EventPreprocessorArrayHisto(int event_input_width, int event_input_height,
                                                           float max_incr_per_pixel,
                                                           float clip_value_after_normalization, bool use_CHW = true,
                                                           float width_scale = 1.f, float height_scale = 1.f) {
    return new Metavision::HistoProcessor<ArrayIt>(event_input_width, event_input_height, max_incr_per_pixel,
                                                   clip_value_after_normalization, use_CHW, width_scale, height_scale);
}

EventPreprocessorArray *create_EventPreprocessorArrayEventCube(Metavision::timestamp delta_t, int event_input_width,
                                                               int event_input_height, int num_utbins,
                                                               bool split_polarity, float max_incr_per_pixel,
                                                               float clip_value_after_normalization = 0.f,
                                                               float width_scale = 1.f, float height_scale = 1.f) {
    return new Metavision::EventCubeProcessor<ArrayIt>(delta_t, event_input_width, event_input_height, num_utbins,
                                                       split_polarity, max_incr_per_pixel,
                                                       clip_value_after_normalization, width_scale, height_scale);
}

EventPreprocessorArray *create_EventPreprocessorArrayHardwareDiff(int sensor_width, int sensor_height, int8_t min_val,
                                                                  int8_t max_val, bool allow_rollover = true) {
    return new Metavision::HardwareDiffProcessor<ArrayIt>(sensor_width, sensor_height, min_val, max_val,
                                                          allow_rollover);
}

EventPreprocessorArray *create_EventPreprocessorArrayHardwareHisto(int sensor_width, int sensor_height,
                                                                   uint8_t neg_saturation = 255,
                                                                   uint8_t pos_saturation = 255) {
    return new Metavision::HardwareHistoProcessor<ArrayIt>(sensor_width, sensor_height, neg_saturation, pos_saturation);
}

EventPreprocessorArray *create_EventPreprocessorArrayTimeSurface(int sensor_width, int sensor_height,
                                                                 bool split_polarity = true) {
    if (split_polarity)
        return new Metavision::TimeSurfaceProcessorSplitPolarities<ArrayIt>(sensor_width, sensor_height);
    else
        return new Metavision::TimeSurfaceProcessorMergePolarities<ArrayIt>(sensor_width, sensor_height);
}

// End Helpers factory

} // namespace

namespace Metavision {

void export_event_preprocessor(py::module &m) {
    py::class_<EventPreprocessorArray>(m, "EventPreprocessor", pybind_doc_core["Metavision::EventPreprocessor"])
        .def_static("create_DiffProcessor", &create_EventPreprocessorArrayDiff, py::arg("input_event_width"),
                    py::arg("input_event_height"), py::arg("max_incr_per_pixel") = 5,
                    py::arg("clip_value_after_normalization") = 1.0, py::arg("scale_width") = 1.f,
                    py::arg("scale_height") = 1.f, pybind_doc_core["Metavision::DiffProcessor::DiffProcessor"])
        .def_static("create_HistoProcessor", &create_EventPreprocessorArrayHisto, py::arg("input_event_width"),
                    py::arg("input_event_height"), py::arg("max_incr_per_pixel") = 5,
                    py::arg("clip_value_after_normalization") = 1.0, py::arg("use_CHW") = true,
                    py::arg("scale_width") = 1.f, py::arg("scale_height") = 1.f,
                    pybind_doc_core["Metavision::HistoProcessor::HistoProcessor"])
        .def_static("create_EventCubeProcessor", &create_EventPreprocessorArrayEventCube, py::arg("delta_t"),
                    py::arg("input_event_width"), py::arg("input_event_height"), py::arg("num_utbins"),
                    py::arg("split_polarity"), py::arg("max_incr_per_pixel") = 63.75,
                    py::arg("clip_value_after_normalization") = 1.0, py::arg("scale_width") = 1.f,
                    py::arg("scale_height") = 1.f,
                    pybind_doc_core["Metavision::EventCubeProcessor::EventCubeProcessor"])
        .def_static("create_HardwareDiffProcessor", &create_EventPreprocessorArrayHardwareDiff,
                    py::arg("input_event_width"), py::arg("input_event_height"), py::arg("min_val"), py::arg("max_val"),
                    py::arg("allow_rollover") = true,
                    pybind_doc_core["Metavision::HardwareDiffProcessor::HardwareDiffProcessor"])
        .def_static("create_HardwareHistoProcessor", &create_EventPreprocessorArrayHardwareHisto,
                    py::arg("input_event_width"), py::arg("input_event_height"), py::arg("neg_saturation") = 255,
                    py::arg("pos_saturation") = 255,
                    pybind_doc_core["Metavision::HardwareHistoProcessor::HardwareHistoProcessor"])
        .def_static("create_TimeSurfaceProcessor", &create_EventPreprocessorArrayTimeSurface,
                    py::arg("input_event_width"), py::arg("input_event_height"), py::arg("split_polarity") = true,
                    R"doc(
                                Creates a TimeSurfaceProcessor instance.

                                    :input_event_width: Width of the input event stream.
                                    :input_event_height: Height of the input event stream.
                                    :split_polarity: (optional) If True, polarities will be managed separately in the
                                                     TimeSurface. Else, a single channel will be used for both
                                                     polarities.
                                )doc")
        .def("init_output_tensor", &EventPreprocessorArray_init_output_tensor, py::return_value_policy::move)
        .def("process_events", &EventPreprocessorArray_process_events_array, py::arg("cur_frame_start_ts"),
             py::arg("events_np"), py::arg("frame_tensor_np"),
             "Takes a chunk of events (numpy array of EventCD) and updates the frame_tensor (numpy array of float)")
        .def(
            "get_frame_size",
            [](const EventPreprocessorArray &cdproc) { return cdproc.get_output_shape().get_nb_values(); },
            "Returns the number of values in the output frame.")
        .def(
            "get_frame_width",
            [](const EventPreprocessorArray &cdproc) { return get_dim(cdproc.get_output_shape(), "W"); },
            "Returns the width of the output frame.")
        .def(
            "get_frame_height",
            [](const EventPreprocessorArray &cdproc) { return get_dim(cdproc.get_output_shape(), "H"); },
            "Returns the height of the output frame.")
        .def(
            "get_frame_channels",
            [](const EventPreprocessorArray &cdproc) { return get_dim(cdproc.get_output_shape(), "C"); },
            "Returns the number of channels of the output frame.")
        .def(
            "get_frame_shape",
            [](const EventPreprocessorArray &cdproc) {
                const auto &dimensions = cdproc.get_output_shape().dimensions;
                std::vector<size_t> shape;
                for (const auto &dim : dimensions)
                    shape.push_back(dim.dim);
                return shape;
            },
            "Returns the frame shape.")
        .def(
            "is_CHW",
            [](const EventPreprocessorArray &cdproc) {
                const auto &dimensions = cdproc.get_output_shape().dimensions;
                return dimensions[0].name == "C" && dimensions[1].name == "H" && dimensions[2].name == "W";
            },
            "Returns true if the output tensor shape has CHW layout.");
}

} // namespace Metavision
