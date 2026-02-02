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

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/pybind/sync_algorithm_process_helper.h"
#include "metavision/sdk/core/algorithms/roi_mask_algorithm.h"

#include "pb_doc_core.h"

namespace py = pybind11;

namespace Metavision {

namespace { // anonymous namsepace
cv::Mat np2cvmat(const py::array_t<double> &np_array) {
    auto info = np_array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("");
    }
    auto nelem = static_cast<size_t>(info.size);
    auto *ptr  = static_cast<double *>(info.ptr);
    auto rows  = static_cast<size_t>(info.shape[0]);
    auto cols  = static_cast<size_t>(info.shape[1]);
    assert(nelem == rows * cols);

    cv::Mat mat(rows, cols, CV_64FC1);
    memcpy(mat.data, ptr, rows * cols * sizeof(double));
    return mat;
}

py::array_t<double> cvmat2np(const cv::Mat &mat) {
    if (!mat.isContinuous()) {
        throw std::invalid_argument("Error: mat buffer is not contiguous");
    }
    if (mat.channels() != 1) {
        std::ostringstream oss;
        oss << "Error: expecting image with only one channel (received: " << mat.channels() << " channels)";
        throw std::invalid_argument(oss.str());
    }
    if (mat.type() != CV_64FC1) {
        throw std::invalid_argument("Error: expecting a CV_64FC1 type cv::mat");
    }
    double *mat_ptr = reinterpret_cast<double *>(mat.data);
    py::array_t<double> np_array(mat.rows * mat.cols, mat_ptr);
    np_array.resize({mat.rows, mat.cols});
    return np_array;
}

void set_pixel_mask_helper(RoiMaskAlgorithm &algo, const py::array_t<double> &pixel_mask_np) {
    const cv::Mat &pixel_mask_cv = np2cvmat(pixel_mask_np);
    algo.set_pixel_mask(pixel_mask_cv);
}

py::array_t<double> get_pixel_mask_helper(RoiMaskAlgorithm &algo) {
    const cv::Mat &pixel_mask_cv = algo.pixel_mask();
    return cvmat2np(pixel_mask_cv);
}

} // end of anonymous namespace

void export_roi_mask_algorithm(py::module &m) {
    py::class_<RoiMaskAlgorithm>(m, "RoiMaskAlgorithm", pybind_doc_core["Metavision::RoiMaskAlgorithm"])
        .def(py::init([](const py::array_t<double> &pixel_mask_np) {
                 const cv::Mat &pixel_mask_cv = np2cvmat(pixel_mask_np);
                 return new RoiMaskAlgorithm(pixel_mask_cv);
             }),
             py::arg("pixel_mask"), pybind_doc_core["Metavision::RoiMaskAlgorithm::RoiMaskAlgorithm"])
        .def("process_events", &process_events_array_sync<RoiMaskAlgorithm, EventCD>, py::arg("input_np"),
             py::arg("output_buf"), doc_process_events_array_sync_str)
        .def("process_events", &process_events_buffer_sync<RoiMaskAlgorithm, EventCD>, py::arg("input_buf"),
             py::arg("output_buf"), doc_process_events_buffer_sync_str)
        .def("process_events_", &process_events_buffer_sync_inplace<RoiMaskAlgorithm, EventCD>, py::arg("events_buf"),
             doc_process_events_buffer_sync_inplace_str)
        .def_static("get_empty_output_buffer", &getEmptyPODBuffer<EventCD>, doc_get_empty_output_buffer_str)
        .def("enable_rectangle", &RoiMaskAlgorithm::enable_rectangle, py::arg("x0"), py::arg("y0"), py::arg("x1"),
             py::arg("y1"), pybind_doc_core["Metavision::RoiMaskAlgorithm::enable_rectangle"])
        .def("set_pixel_mask", &set_pixel_mask_helper, py::arg("mask"),
             pybind_doc_core["Metavision::RoiMaskAlgorithm::set_pixel_mask"])
        .def("pixel_mask", &get_pixel_mask_helper, pybind_doc_core["Metavision::RoiMaskAlgorithm::pixel_mask"])
        .def("max_height", &RoiMaskAlgorithm::max_height, pybind_doc_core["Metavision::RoiMaskAlgorithm::max_height"])
        .def("max_width", &RoiMaskAlgorithm::max_width, pybind_doc_core["Metavision::RoiMaskAlgorithm::max_width"]);
}
} // namespace Metavision