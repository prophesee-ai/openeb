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

#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/utils/pybind/py_array_to_cv_mat.h"

#include "pb_doc_core.h"

namespace Metavision {

namespace {
py::array_t<timestamp> numpy_time_surface_helper(MostRecentTimestampBuffer &time_surface, bool copy = false) {
    std::vector<py::ssize_t> shape = {time_surface.rows(), time_surface.cols(), time_surface.channels()};
    if (time_surface.empty() || copy)
        return py::array_t<timestamp>(shape, time_surface.ptr());

    auto capsule = py::capsule(time_surface.ptr(), [](void *v) {});
    return py::array_t<timestamp>(shape, time_surface.ptr(), capsule);
}

py::buffer_info buffer_info_time_surface_helper(MostRecentTimestampBuffer &time_surface) {
    const py::ssize_t ndim         = (time_surface.channels() == 1 ? 2 : 3);
    std::vector<py::ssize_t> shape = {time_surface.rows(), time_surface.cols()};
    std::vector<py::ssize_t> strides; // stride (in bytes) for each index

    if (time_surface.channels() == 1) {
        strides = {static_cast<py::ssize_t>(sizeof(timestamp) * time_surface.cols()),
                   static_cast<py::ssize_t>(sizeof(timestamp))};
    } else {
        shape.emplace_back(time_surface.channels());
        strides = {static_cast<py::ssize_t>(sizeof(timestamp) * time_surface.cols() * time_surface.rows()),
                   static_cast<py::ssize_t>(sizeof(timestamp) * time_surface.cols()),
                   static_cast<py::ssize_t>(sizeof(timestamp))};
    }

    return py::buffer_info(time_surface.ptr(),                         // pointer to buffer
                           sizeof(timestamp),                          // size of one element
                           py::format_descriptor<timestamp>::format(), // python struct-style format descriptor
                           ndim,                                       // number of dimensions
                           shape,                                      // shape
                           strides);                                   // stride
}

void generate_img_time_surface_helper(MostRecentTimestampBuffer &time_surface, timestamp last_ts, timestamp delta_t,
                                      py::array &image) {
    cv::Mat img_cv;
    Metavision::py_array_to_cv_mat(image, img_cv, false);
    time_surface.generate_img_time_surface(last_ts, delta_t, img_cv);
}

void generate_img_time_surface_collapsing_channels_helper(MostRecentTimestampBuffer &time_surface, timestamp last_ts,
                                                          timestamp delta_t, py::array &image) {
    cv::Mat img_cv;
    Metavision::py_array_to_cv_mat(image, img_cv, false);
    time_surface.generate_img_time_surface_collapsing_channels(last_ts, delta_t, img_cv);
}
} // anonymous namespace

void export_mostrecent_timestamp_buffer(py::module &m) {
    using namespace pybind11::literals;

    py::class_<MostRecentTimestampBuffer, std::shared_ptr<MostRecentTimestampBuffer>>(
        m, "MostRecentTimestampBuffer", py::buffer_protocol(),
        pybind_doc_core["Metavision::MostRecentTimestampBufferT"])
        .def_buffer(&buffer_info_time_surface_helper)
        .def(py::init<int, int, int>(), "rows"_a, "cols"_a, "channels"_a = 1,
             pybind_doc_core["Metavision::MostRecentTimestampBufferT::MostRecentTimestampBufferT(int rows, int cols, "
                             "int channels=1)"])
        .def("numpy", &numpy_time_surface_helper, "Converts to a numpy array", "copy"_a = false)
        .def("_buffer_info", &buffer_info_time_surface_helper)
        .def("set_to", &MostRecentTimestampBuffer::set_to, "ts"_a,
             pybind_doc_core["Metavision::MostRecentTimestampBufferT::set_to"])
        .def("max_across_channels_at", &MostRecentTimestampBuffer::max_across_channels_at, "y"_a, "x"_a,
             pybind_doc_core["Metavision::MostRecentTimestampBufferT::max_across_channels_at"])
        .def_property_readonly("rows", &MostRecentTimestampBuffer::rows,
                               pybind_doc_core["Metavision::MostRecentTimestampBufferT::rows"])
        .def_property_readonly("cols", &MostRecentTimestampBuffer::cols,
                               pybind_doc_core["Metavision::MostRecentTimestampBufferT::cols"])
        .def_property_readonly("channels", &MostRecentTimestampBuffer::channels,
                               pybind_doc_core["Metavision::MostRecentTimestampBufferT::channels"])
        .def("generate_img_time_surface", &generate_img_time_surface_helper, "last_ts"_a, "delta_t"_a, "out"_a,
             pybind_doc_core["Metavision::MostRecentTimestampBufferT::generate_img_time_surface"])
        .def("generate_img_time_surface_collapsing_channels", &generate_img_time_surface_collapsing_channels_helper,
             "last_ts"_a, "delta_t"_a, "out"_a,
             pybind_doc_core["Metavision::MostRecentTimestampBufferT::generate_img_time_surface_collapsing_channels"]);
}

} // namespace Metavision
